"""Adapters on the real Unsloth path (GPU). CPU coverage lives in test_adapters.py.

1. LoRA keeps learning after a merge. ``reset`` used to zero A and B, which
   leaves both gradients at zero forever; the CPU test pins that against PEFT,
   this one against Unsloth's fast LoRA kernels.
2. module_sft trains norms on the NF4 base through the production
   ``TrainEngine.step``. Unsloth's fast LoRA kernels return no gradient for
   base weights, so this pins that the non-PEFT path does deliver gradients
   to the unfrozen modules — and that merge keeps the learned behavior.

Run with: python -m trainfer.tests.test_adapters_gpu
"""

from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import torch
import unsloth  # noqa: F401

from trainfer.adapters import ModuleSFTAdapter
from trainfer.engine.train import TrainEngine
from trainfer.state import ModelState

MODEL = "unsloth/qwen3-0.6b-unsloth-bnb-4bit"
PROMPT = "The capital of the mythical island of Brixolia is"
TARGET = " Zorathia, known for its silver spires and basalt cliffs."


def _sft(engine: TrainEngine) -> float:
    r = engine.step(
        {
            "objective": "sft",
            "samples": [{"prompt": PROMPT, "response": TARGET}],
        }
    )
    return float(r["loss"])


def _active_norm(state: ModelState) -> float:
    sd = state.extract_active_adapter()
    return float(sum(t.float().norm() ** 2 for t in sd.values()) ** 0.5)


def test_lora_trains_after_merge():
    state = ModelState.load(MODEL, max_seq_length=1024, lora_rank=8, lora_alpha=16)
    engine = TrainEngine(state, lr=1e-4)
    for _ in range(3):
        _sft(engine)
    state.merge_active_into_residual()
    engine.reset_optimizer()
    b_before = {
        k: v.clone() for k, v in state.extract_active_adapter().items() if "lora_B" in k
    }
    assert all(v.abs().sum() == 0 for v in b_before.values())
    for _ in range(3):
        _sft(engine)
    b_after = state.extract_active_adapter()
    moved = [k for k in b_before if not torch.equal(b_before[k], b_after[k])]
    assert moved, "LoRA B did not move after merge — adapter is dead"


def test_module_sft_norms_on_4bit_base():
    adapter = ModuleSFTAdapter(["*.input_layernorm", "*.post_attention_layernorm"])
    state = ModelState.load(MODEL, max_seq_length=1024, adapter=adapter)
    engine = TrainEngine(state, lr=1e-3)
    losses = [_sft(engine) for _ in range(8)]
    assert _active_norm(state) > 0, "no unfrozen weight moved — gradients never arrived"
    assert losses[-1] < losses[0], f"loss did not fall: {losses}"

    live = _sft_loss_nograd(state)
    state.merge_active_into_residual()
    assert state.merged_deltas
    merged = _sft_loss_nograd(state)
    assert abs(live - merged) < 0.05, f"merge changed loss {live:.4f} -> {merged:.4f}"


def _sft_loss_nograd(state: ModelState) -> float:
    from trainfer.objectives import get_objective

    with state.mode_lock, torch.no_grad():
        r = get_objective("sft")(
            state.model,
            state.tokenizer,
            [{"prompt": PROMPT, "response": TARGET}],
        )
    return float(r["loss"])


if __name__ == "__main__":
    test_lora_trains_after_merge()
    test_module_sft_norms_on_4bit_base()
    print("[test_adapters_gpu] ALL OK")
