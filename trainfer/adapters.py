"""Adapters: what is trainable on the live model, and how it folds into the residual.

``ModelState`` holds ``base ⊕ merged_deltas ⊕ active``. An :class:`Adapter`
owns the *active* part and everything that depends on its shape:

  * ``attach``          — make the model trainable (wrap in PEFT, unfreeze modules, …).
  * ``reset``           — return the active part to its identity point (live == base ⊕ residual).
  * ``state_dict`` /
    ``load_state_dict`` — snapshot I/O for the active part.
  * ``delta_weights``   — the active part expressed as additive per-weight deltas;
                          ``ModelState.merge_active_into_residual`` accumulates these.
  * ``bind_residual`` /
    ``clear_residual``  — make the merged residual live at forward time.
  * ``reference``       — context manager under which the model runs as the
                          reference policy (base ⊕ residual, active part off).
                          KL anchors, KTO and CCD use this.

Two adapters ship:

``lora``
    PEFT LoRA via Unsloth's ``get_peft_model``. Deltas come from PEFT's own
    ``get_delta_weight`` so rsLoRA scaling is honored. The residual is bound as
    a forward-time sidecar (``W._residual_delta``, read by the patched
    ``matmul_lora`` in ``state.py``) because the NF4 base cannot absorb it.

``module_sft``
    Plain fine-tuning of selected modules (fnmatch patterns over module names).
    Parameters must be floating point — a 4-bit base's quantized linears cannot
    be trained, so target norms / ``lm_head`` / embeddings, or load with
    ``load_in_4bit=False``. The residual is written into the weights in place
    (``W = base + residual + active``) since these weights are not quantized,
    which also makes it correct for non-linear parameters like norm scales.

Methods whose merge is not an additive weight delta (DoRA, IA³) or that have
no weight delta at all (prefix / prompt tuning) do not fit this contract
without extending it.
"""

from __future__ import annotations

import contextlib
import fnmatch
import logging
import math
from typing import Any, ContextManager, Iterator

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Attribute on the live model that points back at its adapter, so objectives
# holding only ``model`` can reach ``reference()``.
_MODEL_ATTR = "_trainfer_adapter"


class Adapter:
    """Base class. Subclasses set ``kind`` and implement every method."""

    kind: str = ""

    def attach(self, model: Any) -> Any:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """JSON-able ``{"kind": ..., **config}`` — stamped into snapshot manifests."""
        raise NotImplementedError

    def reset(self, model: Any) -> None:
        raise NotImplementedError

    def state_dict(self, model: Any) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def load_state_dict(self, model: Any, sd: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def delta_weights(self, model: Any) -> dict[str, torch.Tensor]:
        """Residual key -> fp32 additive delta of the active part. Omit zero deltas."""
        raise NotImplementedError

    def bind_residual(
        self, model: Any, deltas: dict[str, torch.Tensor], hooks: dict[str, Any]
    ) -> None:
        raise NotImplementedError

    def clear_residual(self, model: Any, hooks: dict[str, Any]) -> None:
        raise NotImplementedError

    def reference(self, model: Any) -> ContextManager[Any] | None:
        """Context manager for the reference policy, or None if unsupported."""
        raise NotImplementedError


def tag_model(model: Any, adapter: Adapter) -> None:
    setattr(model, _MODEL_ATTR, adapter)


def model_adapter(model: Any) -> Adapter | None:
    adapter = getattr(model, _MODEL_ATTR, None)
    return adapter if isinstance(adapter, Adapter) else None


def reference_context(model: Any) -> ContextManager[Any] | None:
    """The reference-policy context for ``model``, or None if it has none.

    Prefers the attached adapter; falls back to PEFT's ``disable_adapter()``
    for models built outside ``ModelState.load`` (tests, raw PEFT models).
    """
    adapter = model_adapter(model)
    if adapter is not None:
        return adapter.reference(model)
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return None


def has_reference(model: Any) -> bool:
    adapter = model_adapter(model)
    if adapter is not None:
        return adapter.reference(model) is not None
    return hasattr(model, "disable_adapter")


# --------------------------------------------------------------------- LoRA


def _iter_lora_layers(model: Any) -> Iterator[tuple[str, Any]]:
    for name, module in model.named_modules():
        # PEFT LoraLayer exposes lora_A / lora_B as ModuleDicts.
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if "default" not in module.lora_A:
            continue
        yield name, module


def _make_residual_forward_hook(delta_gpu: torch.Tensor):
    """Closure-scoped forward hook adding ``F.linear(x, delta)`` to base_layer output.

    ``delta_gpu`` is pre-cast to the model compute dtype by
    ``LoraAdapter.bind_residual``; the per-forward .to() only fires on a true
    mismatch (e.g. a layer in fp16 while the residual is bf16).
    """

    def _hook(module: Any, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        delta = delta_gpu
        if delta.dtype != output.dtype or delta.device != output.device:
            delta = delta.to(dtype=output.dtype, device=output.device)
        return output + F.linear(x, delta)

    return _hook


class LoraAdapter(Adapter):
    kind = "lora"

    def __init__(
        self,
        rank: int = 16,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: list[str] | tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES,
        use_rslora: bool = False,
        init: str = "default",
    ) -> None:
        if init not in ("default", "gaussian"):
            # PiSSA / OLoRA / LoftQ inits rewrite the base weight, which the
            # NF4 base + residual scheme cannot represent.
            raise ValueError(
                f"unsupported LoRA init {init!r}; use 'default' or 'gaussian'"
            )
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = list(target_modules)
        self.use_rslora = use_rslora
        self.init = init

    def _peft_init(self) -> bool | str:
        return True if self.init == "default" else self.init

    def attach(self, model: Any) -> Any:
        from unsloth import FastLanguageModel  # heavy import; lazy

        return FastLanguageModel.get_peft_model(
            model,
            r=self.rank,
            target_modules=self.target_modules,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            use_rslora=self.use_rslora,
            init_lora_weights=self._peft_init(),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "use_rslora": self.use_rslora,
            "init": self.init,
        }

    def _scale(self, module: Any) -> float:
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and "default" in scaling:
            return float(scaling["default"])
        if self.use_rslora:
            return self.alpha / math.sqrt(self.rank)
        return self.alpha / self.rank

    def reset(self, model: Any) -> None:
        """Re-initialize every LoRA site: A random, B zero.

        B·A is zero afterwards, so the live model equals base ⊕ residual —
        but unlike zeroing both matrices, gradients still flow (dL/dB ∝ A).
        """
        with torch.no_grad():
            for _name, module in _iter_lora_layers(model):
                if hasattr(module, "reset_lora_parameters"):
                    module.reset_lora_parameters("default", self._peft_init())
                    continue
                A = module.lora_A["default"].weight
                if self.init == "gaussian":
                    torch.nn.init.normal_(A, std=1 / A.shape[0])
                else:
                    torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                module.lora_B["default"].weight.zero_()
        log.info("active LoRA adapter re-initialized (B=0)")

    def state_dict(self, model: Any) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                out[name] = p.detach().to("cpu", dtype=torch.bfloat16).clone()
        return out

    def load_state_dict(self, model: Any, sd: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            named = dict(model.named_parameters())
            for name, t in sd.items():
                if name in named:
                    named[name].data.copy_(
                        t.to(named[name].dtype).to(named[name].device)
                    )

    def delta_weights(self, model: Any) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, module in _iter_lora_layers(model):
                A = module.lora_A["default"].weight  # (r, in)
                B = module.lora_B["default"].weight  # (out, r)
                if B.abs().sum() == 0:
                    continue
                if hasattr(module, "get_delta_weight"):
                    delta = module.get_delta_weight("default")
                else:
                    delta = B.float() @ A.float() * self._scale(module)
                out[f"{name}.weight"] = delta.to(torch.float32)  # (out, in)
        return out

    def bind_residual(
        self, model: Any, deltas: dict[str, torch.Tensor], hooks: dict[str, Any]
    ) -> None:
        """Bind the CPU bf16 residual onto each live LoRA site for forward-time use.

        Two channels:
          (a) ``base_layer.weight._residual_delta = delta_gpu`` — picked up by
              the patched matmul_lora kernel on every QKV/MLP forward under
              Unsloth's fast path.
          (b) ``forward_hook`` on ``base_layer`` that adds
              ``F.linear(x, delta)`` to the layer's output. Fires under the
              standard PEFT path (e.g. during ``model.disable_adapter()`` used
              by the KL anchor in CCPD v2).

        Both channels reference the same delta tensor; if one fires, the other
        doesn't see the layer at all. Under Unsloth fast path, channel (a) fires
        and the hook stays idle. Under PEFT-standard, the hook fires and
        matmul_lora isn't called.
        """
        if not deltas:
            return
        device = next(model.parameters()).device
        applied = 0
        for name, module in _iter_lora_layers(model):
            delta_cpu = deltas.get(f"{name}.weight")
            if delta_cpu is None:
                continue
            delta_gpu = delta_cpu.to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
            base_layer = getattr(module, "base_layer", None)
            if base_layer is None:
                continue
            # Channel (a): attach to the Parameter directly.
            base_layer.weight._residual_delta = delta_gpu  # type: ignore[attr-defined]
            # Channel (b): register forward hook (dedup via stored handle).
            old = hooks.pop(name, None)
            if old is not None:
                try:
                    old.remove()
                except Exception:
                    pass
            hooks[name] = base_layer.register_forward_hook(
                _make_residual_forward_hook(delta_gpu)
            )
            applied += 1
        log.info(
            "applied residual to %d LoRA sites (matmul_lora + forward_hook)", applied
        )

    def clear_residual(self, model: Any, hooks: dict[str, Any]) -> None:
        for handle in list(hooks.values()):
            try:
                handle.remove()
            except Exception:
                pass
        hooks.clear()
        for _name, module in model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            base_layer = getattr(module, "base_layer", None)
            target_w = base_layer.weight if base_layer is not None else None
            if target_w is not None and hasattr(target_w, "_residual_delta"):
                try:
                    del target_w._residual_delta
                except AttributeError:
                    pass
        log.info("cleared residual from live model (hooks + _residual_delta)")

    def reference(self, model: Any) -> ContextManager[Any] | None:
        if hasattr(model, "disable_adapter"):
            return model.disable_adapter()
        return None


# --------------------------------------------------------------------- module SFT


class ModuleSFTAdapter(Adapter):
    """Full fine-tuning of the parameters under selected modules.

    Keeps two copies per trainable parameter: the pristine base on CPU and the
    *anchor* (base ⊕ residual) on the parameter's device. The live weight is
    ``anchor + active``; ``reset`` copies the anchor back, ``reference`` swaps
    it in for the duration of a forward.

    Keys are the model's own parameter names, so tied weights (``lm_head`` ↔
    ``embed_tokens``) appear once and train together.
    """

    kind = "module_sft"

    def __init__(self, modules: list[str] | tuple[str, ...]) -> None:
        if not modules:
            raise ValueError("module_sft needs at least one module pattern")
        self.patterns = list(modules)
        self._params: dict[str, torch.nn.Parameter] = {}
        self._base: dict[str, torch.Tensor] = {}
        self._anchor: dict[str, torch.Tensor] = {}

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "modules": self.patterns}

    def attach(self, model: Any) -> Any:
        modules = dict(model.named_modules())
        matched = [
            n
            for n in modules
            if n and any(fnmatch.fnmatch(n, pat) for pat in self.patterns)
        ]
        if not matched:
            sample = [n for n in modules if n][:8]
            raise ValueError(
                f"module_sft patterns {self.patterns} matched no modules "
                f"(module names look like {sample})"
            )
        wanted = {id(p) for n in matched for p in modules[n].parameters()}
        for p in model.parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if id(p) not in wanted:
                continue
            if not p.is_floating_point():
                raise ValueError(
                    f"module_sft cannot train {name!r} (dtype {p.dtype}): quantized "
                    "weights have no gradient. Load with load_in_4bit=False or "
                    "target non-quantized modules (norms, lm_head, embeddings)."
                )
            p.requires_grad_(True)
            self._params[name] = p
            self._base[name] = p.detach().to("cpu", copy=True)
            self._anchor[name] = p.detach().clone()
        # Frozen embeddings + gradient checkpointing would otherwise cut the
        # graph before any checkpointed layer's trainable params.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        n_elem = sum(p.numel() for p in self._params.values())
        log.info(
            "module_sft: %d modules matched, %d params (%.1fM elements) trainable",
            len(matched),
            len(self._params),
            n_elem / 1e6,
        )
        return model

    def reset(self, model: Any) -> None:
        with torch.no_grad():
            for name, p in self._params.items():
                p.copy_(self._anchor[name])
        log.info("module_sft: active weights reset to anchor")

    def state_dict(self, model: Any) -> dict[str, torch.Tensor]:
        # Active delta, fp32: bf16 - bf16 is exact in fp32, so a round trip
        # through load_state_dict reproduces the weight bit-for-bit.
        return {
            name: (p.detach().float() - self._anchor[name].float()).cpu()
            for name, p in self._params.items()
        }

    def load_state_dict(self, model: Any, sd: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, delta in sd.items():
                p = self._params.get(name)
                if p is None:
                    continue
                anchor = self._anchor[name]
                p.copy_((anchor.float() + delta.to(anchor.device).float()).to(p.dtype))

    def delta_weights(self, model: Any) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in self._params.items():
                delta = p.detach().float() - self._anchor[name].float()
                if delta.abs().sum() == 0:
                    continue
                out[name] = delta
        return out

    def bind_residual(
        self, model: Any, deltas: dict[str, torch.Tensor], hooks: dict[str, Any]
    ) -> None:
        """Move the anchor to base ⊕ ``deltas``, carrying the active part along."""
        unknown = sorted(set(deltas) - set(self._params))
        if unknown:
            raise ValueError(
                f"residual has keys outside the module_sft trainable set: {unknown[:5]}"
            )
        with torch.no_grad():
            for name, p in self._params.items():
                active = p.float() - self._anchor[name].float()
                anchor = self._base[name].to(p.device).float()
                if name in deltas:
                    anchor = anchor + deltas[name].to(p.device).float()
                anchor = anchor.to(p.dtype)
                p.copy_((anchor.float() + active).to(p.dtype))
                self._anchor[name] = anchor
        log.info("module_sft: residual written into %d params", len(deltas))

    def clear_residual(self, model: Any, hooks: dict[str, Any]) -> None:
        self.bind_residual(model, {}, hooks)

    def reference(self, model: Any) -> ContextManager[Any] | None:
        return self._swap_in_anchor()

    @contextlib.contextmanager
    def _swap_in_anchor(self) -> Iterator[None]:
        # Swap storage with ``.data`` rather than ``copy_``: an in-place copy
        # bumps the parameter's version counter, and a live forward that has
        # already saved it for backward would then refuse to backprop.
        saved = {name: p.data for name, p in self._params.items()}
        try:
            for name, p in self._params.items():
                p.data = self._anchor[name]
            yield
        finally:
            for name, p in self._params.items():
                p.data = saved[name]


# --------------------------------------------------------------------- factory


def build_adapter(
    kind: str = "lora",
    *,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_target_modules: list[str] | tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES,
    lora_use_rslora: bool = False,
    lora_init: str = "default",
    trainable_modules: list[str] | tuple[str, ...] = (),
) -> Adapter:
    if kind == "lora":
        return LoraAdapter(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
            use_rslora=lora_use_rslora,
            init=lora_init,
        )
    if kind == "module_sft":
        return ModuleSFTAdapter(trainable_modules)
    raise ValueError(f"unknown adapter {kind!r}; expected 'lora' or 'module_sft'")


def adapter_from_config(cfg: Any) -> Adapter:
    return build_adapter(
        cfg.adapter,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
        lora_use_rslora=cfg.lora_use_rslora,
        lora_init=cfg.lora_init,
        trainable_modules=cfg.trainable_modules,
    )
