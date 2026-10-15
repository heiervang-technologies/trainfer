"""Explicit lossy consolidation, staged one projection at a time on CPU/disk.

No dense base or dense residual is allocated on the GPU. All candidates and
rollback weights are staged before the first live write. The caller holds the
model lock across preparation and application, so inference sees one revision.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import time
import uuid

import torch
from safetensors.torch import load_file, save_file

from . import quantized
from .training_state import rng_state, restore_rng


def error_metrics(base, update, candidate, activations=None):
    """Errors relative to the update, not the much larger frozen base.

    Zero update/retained norms produce null ratios instead of misleading scores.
    Activation-weighted error is optional and requires representative inputs.
    """
    base, update, candidate = base.float(), update.float(), candidate.float()
    retained = candidate - base
    error = retained - update
    update_sq = float(update.square().sum())
    retained_sq = float(retained.square().sum())
    error_sq = float(error.square().sum())
    if not all(math.isfinite(value) for value in (update_sq, retained_sq, error_sq)):
        raise ValueError("nonfinite consolidation error statistics")
    result = {
        "update_squared_norm": update_sq,
        "retained_squared_norm": retained_sq,
        "error_squared_norm": error_sq,
        "update_dot_retained": float((update * retained).sum()),
        "relative_update_error": math.sqrt(error_sq / update_sq) if update_sq else None,
        "retained_norm_ratio": math.sqrt(retained_sq / update_sq) if update_sq else None,
        "update_cosine": float((update * retained).sum()) / math.sqrt(update_sq * retained_sq)
        if update_sq and retained_sq else None,
        "unchanged_weight_fraction": float((candidate == base).float().mean()),
    }
    if activations is not None:
        x = activations.detach().cpu().float()
        result["activation_error_squared_norm"] = float((x @ error.T).square().sum())
        result["activation_update_squared_norm"] = float((x @ update.T).square().sum())
    return result


def _candidate(state, module_name, module, base, max_cpu_workspace_bytes):
    a, b = module.lora_A["default"].weight, module.lora_B["default"].weight
    # Conservative working-tensor estimate, not an RSS guarantee (BLAS/backend
    # and allocator workspace are implementation-dependent). Reject before
    # allocating dense CPU matrices if the configured budget is insufficient.
    elements = a.shape[1] * b.shape[0]
    estimated_bytes = elements * 48 + (a.numel() + b.numel()) * 4
    if estimated_bytes > max_cpu_workspace_bytes:
        raise ValueError(f"{module_name} needs an estimated {estimated_bytes} CPU workspace bytes")
    original = quantized.pack(base)
    if getattr(base.weight, "quant_state", None) is not None:
        from bitsandbytes.functional import QuantState, dequantize_4bit, quantize_4bit
    qstate = (
        QuantState.from_dict({k: v for k, v in original.items() if k != "weight"}, device="cpu")
        if getattr(base.weight, "quant_state", None) is not None else None
    )
    weight = dequantize_4bit(original["weight"], qstate).float() if qstate is not None else original["weight"].float()
    scale = getattr(module, "scaling", {}).get("default", state.lora_alpha / state.lora_rank)
    update = (b.detach().cpu().float() @ a.detach().cpu().float()) * scale
    residual = state.merged_deltas.get(module_name + ".weight")
    if residual is not None:
        update.add_(residual.detach().cpu().float())
    if not bool(torch.isfinite(update).all()):
        raise ValueError(f"nonfinite update in {module_name}")
    if not bool(torch.count_nonzero(update)):
        return original, original, error_metrics(weight, update, weight), estimated_bytes
    target = weight + update
    if not bool(torch.isfinite(target).all()):
        raise ValueError(f"nonfinite target in {module_name}")
    if qstate is not None:
        data, quant = quantize_4bit(
            target.to(qstate.dtype), blocksize=qstate.blocksize,
            compress_statistics=qstate.nested, quant_type=qstate.quant_type,
            quant_storage=original["weight"].dtype,
        )
        candidate = dequantize_4bit(data, quant).float()
        result = {"weight": data, **quant.as_dict(packed=True)}
    else:
        # Dynamic 4-bit models deliberately leave sensitive projections dense.
        # Preserve that choice rather than quantizing them incidentally.
        data = target.to(original["weight"].dtype)
        candidate = data.float()
        result = {"weight": data}
    if not bool(torch.isfinite(candidate).all()):
        raise ValueError(f"nonfinite quantized candidate in {module_name}")
    quantized.validate(base, result)
    return original, result, error_metrics(weight, update, candidate), estimated_bytes


def consolidate(state, workspace: Path, *, preview: bool = False,
                max_relative_update_error: float | None = None,
                max_cpu_workspace_bytes: int = 4 * 1024**3):
    """Preview or apply a CPU requantized merge; always opt-in and lossy.

    The error gate applies to the aggregate Frobenius norm, not behavior or task
    accuracy. Layer reports expose concentration of error; callers choose policy.
    """
    if max_relative_update_error is not None and (
        not math.isfinite(max_relative_update_error) or max_relative_update_error < 0
    ):
        raise ValueError("max_relative_update_error must be finite and nonnegative")
    if max_cpu_workspace_bytes < 1:
        raise ValueError("CPU workspace budget must be positive")
    started = time.monotonic()
    with state.mode_lock, torch.no_grad():
        modules = {n: m for n, m in state.model.named_modules() if hasattr(m, "lora_A")}
        bases = quantized.projection_layers(state)
        if not modules or set(modules) != set(bases):
            raise ValueError("consolidation requires a supported base layer for every adapted projection")
        for module in modules.values():
            if set(module.lora_A) != {"default"} or set(module.lora_B) != {"default"}:
                raise ValueError("requantization supports one default adapter")
            if getattr(module, "use_dora", {}).get("default", False):
                raise ValueError("DoRA consolidation is not supported")
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        rows = []
        with tempfile.TemporaryDirectory(prefix="requant-", dir=workspace) as temporary:
            directory = Path(temporary)
            for index, (name, module) in enumerate(modules.items()):
                original, candidate, metrics, cpu_bytes = _candidate(
                    state, name, module, bases[name], max_cpu_workspace_bytes,
                )
                if not preview:
                    save_file(original, str(directory / f"old-{index}.safetensors"))
                    save_file(candidate, str(directory / f"new-{index}.safetensors"))
                rows.append({"layer": name, "estimated_cpu_workspace_bytes": cpu_bytes, **metrics})
                del original, candidate
            update_sq = sum(r["update_squared_norm"] for r in rows)
            error_sq = sum(r["error_squared_norm"] for r in rows)
            relative = math.sqrt(error_sq / update_sq) if update_sq else None
            report = {
                "strategy": "cpu_requantize", "preview": preview, "applied": False,
                "relative_update_error": relative, "layers": rows,
                "estimated_cpu_workspace_bytes": max(r["estimated_cpu_workspace_bytes"] for r in rows),
                "dense_gpu_residual_bytes_after": 0,
                "base_revision": state.base_revision,
                "default_error_gate_passes": relative is None or relative < 1.0,
            }
            # Error >= update norm can erase or outweigh the intended change.
            # This is a necessary geometric screen, not a behavioral guarantee.
            # Preview always reports it; applying with a larger allowance is an
            # explicit caller decision, never an automatic fallback.
            if not preview and max_relative_update_error is None and relative is not None and relative >= 1.0:
                raise ValueError(
                    f"requantization relative update error {relative:.6g} is not smaller than the update; "
                    "defer consolidation or explicitly set max_relative_update_error"
                )
            if max_relative_update_error is not None and relative is not None and relative > max_relative_update_error:
                raise ValueError(f"requantization relative update error {relative:.6g} exceeds limit {max_relative_update_error}")
            if not preview and update_sq:
                previous_adapter = state.extract_active_adapter()
                previous_residual = state.merged_deltas
                previous_rng = rng_state()
                try:
                    for index, name in enumerate(modules):
                        quantized.apply(bases[name], load_file(str(directory / f"new-{index}.safetensors")))
                    state._clear_residual_from_model()
                    state.merged_deltas = {}
                    state.reset_active_adapter()
                except Exception:
                    for index, name in enumerate(modules):
                        quantized.apply(bases[name], load_file(str(directory / f"old-{index}.safetensors")))
                    state.merged_deltas = previous_residual
                    state._apply_residual_to_model()
                    state.load_active_adapter(previous_adapter)
                    restore_rng(previous_rng)
                    raise
                state.base_revision = uuid.uuid4().hex
                state.merges_applied += 1
                report.update(applied=True, base_revision=state.base_revision)
            report["wall_seconds"] = time.monotonic() - started
            return report
