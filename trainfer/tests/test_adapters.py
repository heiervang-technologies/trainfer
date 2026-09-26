"""Adapter contract: reset / merge / snapshot / reference for LoRA and module_sft.

Runs on CPU against a tiny randomly-initialized Llama. The PEFT-backed tests
skip when ``peft`` is not installed; the rest need only torch + transformers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from trainfer.adapters import (
    LoraAdapter,
    ModuleSFTAdapter,
    build_adapter,
    has_reference,
    reference_context,
    tag_model,
)
from trainfer.snapshot import SnapshotManager
from trainfer.state import ModelState

pytestmark = pytest.mark.cpu_only


def _tiny_llama(tie: bool = False):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=tie,
    )
    return transformers.LlamaForCausalLM(cfg).eval()


def _state(model, adapter) -> ModelState:
    tag_model(model, adapter)
    return ModelState(
        model=model,
        tokenizer=None,
        base_model_name="test/tiny",
        lora_rank=4,
        lora_alpha=8,
        _adapter=adapter,
    )


def _ids() -> torch.Tensor:
    return torch.arange(1, 9).unsqueeze(0)


def _logits(model) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=_ids()).logits.float()


def _sgd_step(model, lr: float = 0.5) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    model(input_ids=_ids(), labels=_ids()).loss.backward()
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p -= lr * p.grad


def _peft_state(use_rslora: bool = False) -> ModelState:
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        _tiny_llama(),
        peft.LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj", "down_proj"],
            use_rslora=use_rslora,
        ),
    )
    return _state(model, LoraAdapter(rank=4, alpha=8, use_rslora=use_rslora))


# ------------------------------------------------------------------ LoRA


def test_lora_can_train_after_merge():
    """Regression: reset used to zero A *and* B, so every step after the first
    merge produced zero gradients and the adapter was dead for good."""
    state = _peft_state()
    _sgd_step(state.model)
    state.merge_active_into_residual()
    before = {k: v.clone() for k, v in state.extract_active_adapter().items()}
    _sgd_step(state.model)
    after = state.extract_active_adapter()
    moved = [k for k in before if not torch.equal(before[k], after[k])]
    assert moved, "no LoRA parameter moved after a merge — adapter is dead"


def test_lora_merge_preserves_forward():
    state = _peft_state()
    for _ in range(3):
        _sgd_step(state.model)
    live = _logits(state.model)
    state.merge_active_into_residual()
    assert state.merged_deltas
    merged = _logits(state.model)
    # Residual is bf16; the gap is rounding, not a lost update.
    assert (live - merged).abs().max() < 5e-2
    assert (live - merged).abs().max() < (live - _logits_base(state)).abs().max()


def _logits_base(state: ModelState) -> torch.Tensor:
    saved = dict(state.merged_deltas)
    state.merged_deltas = {}
    state._clear_residual_from_model()
    with reference_context(state.model):
        out = _logits(state.model)
    state.merged_deltas = saved
    state._apply_residual_to_model()
    return out


def test_lora_delta_honors_rslora_scaling():
    state = _peft_state(use_rslora=True)
    for _, module in state.model.named_modules():
        if hasattr(module, "lora_B") and "default" in getattr(module, "lora_B", {}):
            nn.init.normal_(module.lora_B["default"].weight)
    deltas = state.adapter.delta_weights(state.model)
    assert deltas
    name = next(iter(deltas))
    module = dict(state.model.named_modules())[name[: -len(".weight")]]
    A = module.lora_A["default"].weight.float()
    B = module.lora_B["default"].weight.float()
    expected = B @ A * (8 / 4**0.5)
    torch.testing.assert_close(deltas[name], expected)


def test_lora_reset_without_peft_helpers():
    """Fake LoRA layers (no reset_lora_parameters) still get A random, B zero."""

    class FakeLora(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(4, 4, bias=False)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})

    model = nn.Sequential(FakeLora())
    adapter = LoraAdapter(rank=2, alpha=4)
    adapter.reset(model)
    assert model[0].lora_A["default"].weight.abs().sum() > 0
    assert model[0].lora_B["default"].weight.abs().sum() == 0
    assert adapter.delta_weights(model) == {}


def test_lora_rejects_base_rewriting_init():
    with pytest.raises(ValueError, match="init"):
        LoraAdapter(init="pissa")


# ------------------------------------------------------------------ module_sft


def _sft_state(patterns=("model.norm", "lm_head"), tie: bool = False) -> ModelState:
    model = _tiny_llama(tie=tie)
    adapter = ModuleSFTAdapter(list(patterns))
    model = adapter.attach(model)
    return _state(model, adapter)


def test_module_sft_only_matched_params_train():
    state = _sft_state(["model.norm", "*.1.mlp.down_proj"])
    trainable = {n for n, p in state.model.named_parameters() if p.requires_grad}
    assert trainable == {"model.norm.weight", "model.layers.1.mlp.down_proj.weight"}
    frozen_before = state.model.model.layers[0].mlp.down_proj.weight.clone()
    norm_before = state.model.model.norm.weight.clone()
    _sgd_step(state.model)
    assert torch.equal(state.model.model.layers[0].mlp.down_proj.weight, frozen_before)
    assert not torch.equal(state.model.model.norm.weight, norm_before)


def test_module_sft_tied_weights_train_once():
    state = _sft_state(["lm_head"], tie=True)
    trainable = [n for n, p in state.model.named_parameters() if p.requires_grad]
    assert trainable == ["model.embed_tokens.weight"]


def test_module_sft_reference_is_pre_training_and_backward_survives():
    state = _sft_state()
    base = _logits(state.model)
    _sgd_step(state.model)
    trained = _logits(state.model)
    assert not torch.allclose(base, trained)

    # Live forward saves the weights for backward; a reference forward in
    # between must not invalidate them (in-place swaps would bump versions).
    out = state.model(input_ids=_ids(), labels=_ids())
    with torch.no_grad(), reference_context(state.model):
        ref = state.model(input_ids=_ids()).logits.float()
    out.loss.backward()
    torch.testing.assert_close(ref, base)
    torch.testing.assert_close(_logits(state.model), trained)


def test_module_sft_merge_reset_and_snapshot_roundtrip():
    state = _sft_state()
    _sgd_step(state.model)
    trained = _logits(state.model)

    state.merge_active_into_residual()
    assert set(state.merged_deltas) == {"model.norm.weight", "lm_head.weight"}
    assert state.adapter.delta_weights(state.model) == {}
    # bf16 residual: rounding only.
    assert (_logits(state.model) - trained).abs().max() < 5e-2
    merged = _logits(state.model)
    # Reference now anchors at base ⊕ residual.
    with torch.no_grad(), reference_context(state.model):
        torch.testing.assert_close(_logits(state.model), merged)

    _sgd_step(state.model)
    live = _logits(state.model)
    with tempfile.TemporaryDirectory() as td:
        mgr = SnapshotManager(Path(td))
        mgr.save("s", state)
        state.reset_active_adapter()
        torch.testing.assert_close(_logits(state.model), merged)
        manifest = mgr.load("s", state)
    assert manifest["adapter"] == {
        "kind": "module_sft",
        "modules": ["model.norm", "lm_head"],
    }
    torch.testing.assert_close(_logits(state.model), live)


def test_module_sft_snapshot_into_fresh_model():
    state = _sft_state()
    _sgd_step(state.model)
    state.merge_active_into_residual()
    _sgd_step(state.model)
    live = _logits(state.model)
    fresh = _sft_state()
    with tempfile.TemporaryDirectory() as td:
        mgr = SnapshotManager(Path(td))
        mgr.save("s", state)
        mgr.load("s", fresh)
    torch.testing.assert_close(_logits(fresh.model), live)


def test_module_sft_rejects_quantized_and_unmatched():
    model = _tiny_llama()
    with pytest.raises(ValueError, match="matched no modules"):
        ModuleSFTAdapter(["nope.*"]).attach(model)
    model.model.norm.weight = nn.Parameter(
        torch.zeros(16, dtype=torch.uint8), requires_grad=False
    )
    with pytest.raises(ValueError, match="quantized"):
        ModuleSFTAdapter(["model.norm"]).attach(model)


# ------------------------------------------------------------------ plumbing


def test_snapshot_rejects_adapter_kind_mismatch():
    sft = _sft_state()
    lora = _peft_state()
    with tempfile.TemporaryDirectory() as td:
        mgr = SnapshotManager(Path(td))
        mgr.save("lora", lora)
        with pytest.raises(ValueError, match="adapter 'lora'"):
            mgr.load("lora", sft)


def test_reference_falls_back_to_disable_adapter():
    class Peftish(nn.Module):
        def disable_adapter(self):
            return torch.no_grad()

    assert has_reference(Peftish())
    assert reference_context(Peftish()) is not None
    assert not has_reference(nn.Linear(2, 2))
    assert reference_context(nn.Linear(2, 2)) is None


def test_build_adapter_and_cli():
    from trainfer.cli import parse_cli_args

    cfg = parse_cli_args(
        ["--adapter", "module_sft", "--trainable-modules", "model.norm", "lm_head"]
    )
    assert cfg.adapter == "module_sft"
    assert cfg.trainable_modules == ["model.norm", "lm_head"]
    assert isinstance(build_adapter("lora"), LoraAdapter)
    cfg = parse_cli_args(["--rslora", "--lora-target-modules", "q_proj", "v_proj"])
    assert cfg.lora_use_rslora and cfg.lora_target_modules == ["q_proj", "v_proj"]
    with pytest.raises(ValueError):
        build_adapter("dora")
    with pytest.raises(ValueError):
        build_adapter("module_sft")
