"""Live model state: base_weights ⊕ merged_deltas ⊕ active_adapter.

The active adapter is pluggable (``trainfer/adapters.py``): LoRA by default,
or plain fine-tuning of selected modules. The notes below describe the LoRA
path, which is the one that needs the residual patch.

The invariant from LIVELEARN §3.1/§6:
  * base is NF4, never modified.
  * merged_deltas is a bf16 residual held in CPU RAM between merges, applied at
    load time by adding into a dequantized copy of the base (never requantized
    back to NF4 — that's the 4-bit merge gotcha from §6).
  * active_lora_adapter is hot in GPU; gradients flow into it.

For this implementation we rely on Unsloth's FastLanguageModel + PEFT LoRA. The
merged_deltas residual is materialized by Unsloth's existing dequant→fp32→merge
path when `merge()` is called; we keep our own bf16 copy so we can repeat the
merge or restore it on load.

Live residual application
-------------------------
Unsloth's fast path (Qwen3 attention + MLP) bypasses both ``LoraLayer.forward``
and ``base_layer.forward``. A naive forward-hook on either never fires, so a
merged residual would sit inert on CPU. We close that gap by monkey-patching
``unsloth.kernels.utils.matmul_lora`` — the single low-level kernel every
QKV/O/gate/up/down call funnels into. The patch:

  * reads a sidecar attribute ``W._residual_delta`` (a bf16 GPU tensor),
  * adds ``F.linear(X, delta)`` to the kernel's output,
  * is a no-op when the attribute is absent.

Attaching the delta to the Parameter itself (rather than Gemini's
``id(W) -> delta`` global dict) keeps the binding local to the layer and
survives Unsloth's ``for_training``/``for_inference`` mode flips (``id(W)``
stays stable across those flips, empirically verified on Qwen3).

A fallback ``forward_pre_hook`` is also registered on ``base_layer`` so the
standard PEFT path (used under ``model.disable_adapter()`` for the KL anchor)
still applies the residual.
"""

from __future__ import annotations

import functools
import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .adapters import Adapter, LoraAdapter, tag_model

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- matmul_lora patch
# Module-level, installed once on import. Idempotent via the sentinel attribute.
#
# Upstream coupling: our wrapper assumes matmul_lora's signature is
# (X, W, W_quant, A, B, s, out=None). If a future Unsloth refactor changes
# param names or count, we abort the install loudly rather than run with a
# silently-wrong residual — trainfer still works, just without the fast-path
# residual (PEFT-standard forward_pre_hook backstop still fires).
_EXPECTED_MATMUL_LORA_PARAMS = ("X", "W", "W_quant", "A", "B", "s", "out")


def _install_matmul_lora_patch() -> None:
    import inspect
    import unsloth.kernels.utils as _uutils  # heavy but already paid for by Unsloth

    original = _uutils.matmul_lora
    if getattr(original, "_trainfer_patched", False):
        return

    try:
        params = tuple(inspect.signature(original).parameters.keys())
    except (TypeError, ValueError):
        params = ()
    if params != _EXPECTED_MATMUL_LORA_PARAMS:
        log.warning(
            "trainfer: unsloth.kernels.utils.matmul_lora signature changed "
            "(got %s, expected %s) — skipping residual patch install. "
            "Live residual via fast path is DISABLED; PEFT forward_pre_hook "
            "backstop still covers disable_adapter() codepaths.",
            params,
            _EXPECTED_MATMUL_LORA_PARAMS,
        )
        return

    def _patched(X, W, W_quant, A, B, s, out=None):
        out_res = original(X, W, W_quant, A, B, s, out=out)
        delta = getattr(W, "_residual_delta", None)
        if delta is None:
            return out_res
        # Residual is a (out_features, in_features) bf16 GPU tensor, no grad,
        # pre-cast at bind time by ``_apply_residual_to_model`` to match the
        # model compute dtype. Skip the per-forward .to() syscall when the
        # dtype/device already match (the common path on bf16 pipelines).
        if delta.dtype != out_res.dtype or delta.device != out_res.device:
            delta = delta.to(
                dtype=out_res.dtype, device=out_res.device, non_blocking=True
            )
        return out_res + F.linear(X, delta)

    _patched._trainfer_patched = True  # type: ignore[attr-defined]
    _patched._trainfer_original = original  # type: ignore[attr-defined]

    # Replace every binding across the unsloth package. Unsloth re-exports
    # matmul_lora in several submodules (kernels.utils, kernels.fast_lora, …)
    # and pre-bound references in those modules must also flip — otherwise the
    # fast path in fast_lora still sees the unpatched callable.
    replaced = 0
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("unsloth"):
            continue
        try:
            members = list(vars(mod).items())
        except Exception:
            continue
        for attr, val in members:
            if val is original:
                setattr(mod, attr, _patched)
                replaced += 1
    log.debug("installed matmul_lora residual patch (%d bindings rewritten)", replaced)


@functools.cache
def install() -> None:
    """Idempotent: patches unsloth's matmul_lora to honor ``W._residual_delta``.

    Called automatically by ``ModelState.load``. External callers can invoke
    ``trainfer.install()`` explicitly to pre-warm the patch before constructing a
    ModelState. Previously this ran as an import-time side effect, which
    meant a plain ``from trainfer.state import ModelState`` (dataclass reads,
    snapshot-manifest inspection, schema probes) mutated
    ``sys.modules["unsloth.*"]``. See PR#8 review.
    """
    _install_matmul_lora_patch()


@dataclass
class ModelState:
    """Owns the live model + tokenizer + adapter.

    What is trainable, how it merges, and how the residual is bound are the
    adapter's business (``trainfer/adapters.py``). ``merged_deltas`` is the
    canonical record either way: key -> bf16 CPU delta.
    """

    model: Any  # FastLanguageModel-wrapped model (PEFT-wrapped for LoRA)
    tokenizer: Any
    base_model_name: str
    lora_rank: int
    lora_alpha: int
    # Bookkeeping of what has been merged so far in this process.
    merges_applied: int = 0
    # CPU bf16 residual: name -> delta tensor from all prior merges.
    merged_deltas: dict[str, torch.Tensor] = field(default_factory=dict)
    # Serializes train/infer mode flips. Unsloth's FastLanguageModel.for_inference
    # and .for_training toggle per-layer temp buffers (temp_QA/temp_O/…); if a
    # train step runs for_training() while a concurrent chat is mid-generate on
    # the inference fast path, the buffers vanish mid-call. Everything that
    # calls .generate() or .backward() must hold this lock across the mode flip
    # AND the work that depends on those buffers.
    mode_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    # Base-layer forward hook handles (LoRA fallback path for disable_adapter).
    # Keyed by LoraLayer name; stored so we can remove/rebind without leaking.
    _residual_hook_handles: dict[str, Any] = field(default_factory=dict, repr=False)
    # Optional frozen reference model (base-only, eval, requires_grad=False).
    # When set, objectives consume it as ``pi_ref`` for KL anchoring instead
    # of falling back to the adapter's reference context on the live model.
    frozen_ref: Any = field(default=None, repr=False)
    _adapter: Adapter | None = field(default=None, repr=False)
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    @property
    def adapter(self) -> Adapter:
        # States built via ``ModelState.__new__`` (tests) never ran __init__;
        # they carry a PEFT-shaped model, so default to LoRA.
        adapter = self.__dict__.get("_adapter")
        if adapter is None:
            adapter = LoraAdapter(rank=self.lora_rank, alpha=self.lora_alpha)
            self._adapter = adapter
        return adapter

    @classmethod
    def load(
        cls,
        model_name: str,
        max_seq_length: int = 2048,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        load_in_4bit: bool = True,
        adapter: Adapter | None = None,
    ) -> "ModelState":
        """Load the base model and attach ``adapter``.

        ``adapter`` defaults to LoRA built from the ``lora_*`` arguments.
        """
        # Install the matmul_lora residual patch before Unsloth finishes
        # binding its fast-path references. Idempotent + cached.
        install()
        from unsloth import FastLanguageModel  # heavy import; lazy

        if adapter is None:
            adapter = LoraAdapter(
                rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
            )
        log.info(
            "loading base model %s (4bit=%s, adapter=%s)",
            model_name,
            load_in_4bit,
            adapter.kind,
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )
        model = adapter.attach(model)
        tag_model(model, adapter)
        return cls(
            model=model,
            tokenizer=tokenizer,
            base_model_name=model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            _adapter=adapter,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

    # ------------------------------------------------------------------ adapter ops
    def reset_active_adapter(self) -> None:
        """Return the active adapter to identity (live == base ⊕ residual)."""
        self.adapter.reset(self.model)

    def extract_active_adapter(self) -> dict[str, torch.Tensor]:
        """Snapshot the trainable part only."""
        return self.adapter.state_dict(self.model)

    def load_active_adapter(self, sd: dict[str, torch.Tensor]) -> None:
        self.adapter.load_state_dict(self.model, sd)

    # ------------------------------------------------------------------ merge
    def merge_active_into_residual(self) -> None:
        """Bake the active adapter into `merged_deltas` (CPU bf16 residual).

        The adapter reports its active part as additive per-weight deltas
        (for LoRA, Δ = scaling · B @ A per site); we accumulate those into
        `merged_deltas`, reset the adapter to identity, and rebind the
        residual. The NF4 base is never touched — this keeps it stable per §6.

        The residual is the canonical record of what has been trained; after
        a merge the active part is identity, so base ⊕ residual == live model.
        """
        new_deltas: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for key, delta in self.adapter.delta_weights(self.model).items():
                prev = self.merged_deltas.get(key)
                if prev is not None:
                    delta = delta + prev.to(delta.device, dtype=torch.float32)
                new_deltas[key] = delta.to("cpu", dtype=torch.bfloat16)
        self.merged_deltas.update(new_deltas)
        self.merges_applied += 1
        self.reset_active_adapter()
        self._apply_residual_to_model()
        log.info(
            "merged %d %s sites; total merges=%d",
            len(new_deltas),
            self.adapter.kind,
            self.merges_applied,
        )

    # ------------------------------------------------------------------ residual → live
    def _hooks(self) -> dict[str, Any]:
        # Tolerates states built via ``__new__`` that never set the field.
        return self.__dict__.setdefault("_residual_hook_handles", {})

    def _clear_residual_from_model(self) -> None:
        """Unbind any residual from the live model.

        Called when a snapshot with empty merged_deltas is loaded over a model
        that previously had a non-empty residual. Without this, the old
        bindings leak into the new state, silently corrupting inference.
        See review finding C-6.
        """
        self.adapter.clear_residual(self.model, self._hooks())

    def _apply_residual_to_model(self) -> None:
        """Make ``merged_deltas`` live at forward time (see the adapter's ``bind_residual``)."""
        if not self.merged_deltas:
            return
        self.adapter.bind_residual(self.model, self.merged_deltas, self._hooks())

    def residual_fingerprint(self) -> str:
        """Deterministic hash of the CPU residual — used by snapshot round-trip tests."""
        import hashlib

        h = hashlib.sha256()
        for k in sorted(self.merged_deltas.keys()):
            t = self.merged_deltas[k]
            h.update(k.encode("utf-8"))
            h.update(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    # ------------------------------------------------------------------ frozen ref
    def load_frozen_ref(self) -> None:
        """Load a second base-only model as the frozen reference for KL anchors.

        The live ``self.model`` carries the active adapter plus any applied
        residual. A frozen reference held in eval mode with grad disabled gives
        objectives a *stable* anchor point — the session-start policy — rather
        than the adapter's reference context, which anchors to the live
        merged-deltas state.

        Memory cost: the 4-bit NF4 base weighs roughly 0.4 GB for a 0.6B model
        and ~5 GB for a 7B model. Call from ``Controller.start`` only when
        ``cfg.frozen_ref=True`` so the VRAM cost is opt-in.
        """
        from unsloth import FastLanguageModel

        if self.frozen_ref is not None:
            log.info("frozen reference already loaded — skipping")
            return
        log.info("loading frozen reference (base-only) from %s", self.base_model_name)
        ref_model, _ = FastLanguageModel.from_pretrained(
            model_name=self.base_model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            dtype=None,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        # Put into inference mode so the Unsloth fast path works.
        try:
            FastLanguageModel.for_inference(ref_model)
        except Exception:
            pass
        self.frozen_ref = ref_model
        log.info("frozen reference loaded (eval, requires_grad=False)")

    def save_residual(self, path: Path) -> None:
        from safetensors.torch import save_file

        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.merged_deltas:
            # safetensors refuses empty; write a sentinel.
            path.with_suffix(".empty").touch()
            if path.exists():
                path.unlink()
            return
        save_file({k: v.contiguous() for k, v in self.merged_deltas.items()}, str(path))

    def load_residual(self, path: Path) -> None:
        from safetensors.torch import load_file

        if not path.exists():
            self.merged_deltas = {}
            return
        self.merged_deltas = load_file(str(path))
        self._apply_residual_to_model()
