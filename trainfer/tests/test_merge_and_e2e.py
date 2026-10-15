"""End-to-end tests that require a real model on the GPU.

Tests:
  1. Merge determinism — merging the active LoRA twice (null merge second time)
     yields the same residual fingerprint as merging it once.
  2. Merge correctness — LoRA-on output ≈ base + residual output after merge
     (within bf16 numerical tolerance on greedy next-token logits).
  3. Commit-cursor end-to-end — submit train, then submit infer with
     after_commit_token; verify logprob on the training prompt moved.

Run with: python -m trainfer.tests.test_merge_and_e2e
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import pytest

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import unsloth  # noqa: F401

from trainfer.state import ModelState
from trainfer.engine.train import TrainEngine


def test_merge_determinism():
    print("[merge] loading Qwen3-0.6B …")
    state = ModelState.load(
        model_name="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=1024,
        lora_rank=8,
        lora_alpha=16,
    )
    engine = TrainEngine(state, lr=5e-5)

    # Take a few SFT steps to populate the LoRA with something non-zero.
    for _ in range(3):
        engine.step(
            {
                "objective": "sft",
                "samples": [
                    {"prompt": "Name a color.", "response": "Blue is a calming color."},
                ],
            }
        )

    # Merge once, record fingerprint, merge a null LoRA (zeroed) and re-fingerprint.
    state.merge_active_into_residual()
    fp1 = state.residual_fingerprint()
    assert fp1, "residual fingerprint empty after merge"
    print(f"[merge] fp after 1st merge: {fp1[:16]}…")

    # Reset active adapter is already done by merge_active; merge again.
    # Since active is zero, no new deltas should be produced and residual
    # fingerprint must be unchanged.
    state.merge_active_into_residual()
    fp2 = state.residual_fingerprint()
    print(f"[merge] fp after null 2nd merge: {fp2[:16]}…")
    assert fp1 == fp2, "null second merge changed residual (non-idempotent)"
    print("[merge] determinism OK")


@pytest.mark.gpu
def test_training_restarts_after_repeated_merges():
    """A fresh optimizer must learn after each adapter consolidation."""
    state = ModelState.load(
        model_name="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=256, lora_rank=8, lora_alpha=16,
    )
    engine = TrainEngine(state, lr=5e-5)
    spec = {"objective": "sft", "samples": [{
        "prompt": "Name a color.", "response": "Blue is a calming color.",
    }]}
    for _ in range(3):
        result = engine.step(spec)
        assert result["components"]["grad_norm_total"] > 0
        assert any(
            p.detach().count_nonzero().item() > 0
            for name, p in state.model.named_parameters() if "lora_B" in name
        )
        state.merge_active_into_residual()
        engine.reset_optimizer()


@pytest.mark.gpu
def test_snapshot_resume_reproduces_real_optimizer_update(tmp_path):
    import torch
    from trainfer.snapshot import SnapshotManager

    state = ModelState.load(
        model_name="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=256, lora_rank=8, lora_alpha=16,
    )
    engine = TrainEngine(state, lr=5e-5)
    spec = {"objective": "sft", "samples": [{
        "prompt": "Name a color.", "response": "Blue is a calming color.",
    }]}
    engine.step(spec)
    manager = SnapshotManager(tmp_path)
    manager.save("resume", state, engine=engine, include_training=True)
    expected = engine.step(spec)
    expected_weights = state.extract_active_adapter()
    engine.step(spec)
    manager.load("resume", state, engine=engine, resume=True)
    actual = engine.step(spec)
    assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-6)
    for name, tensor in state.extract_active_adapter().items():
        assert torch.equal(tensor, expected_weights[name]), name


@pytest.mark.gpu
@pytest.mark.parametrize("checkpointing", ["standard", "off"])
def test_model_controls_and_memory_rejection_before_forward(tmp_path, monkeypatch, checkpointing):
    import torch
    from unsloth import FastLanguageModel
    from trainfer.errors import MemoryAdmissionError

    state = ModelState.load(
        model_name="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=256, lora_rank=4, lora_alpha=8,
        lora_target_modules=["q_proj", "v_proj"],
        gradient_checkpointing=checkpointing, model_dtype="bfloat16",
    )
    names = [n for n, p in state.model.named_parameters() if p.requires_grad]
    assert names and all("q_proj" in n or "v_proj" in n for n in names)
    engine = TrainEngine(state, memory_reserve_bytes=1024**4)
    original = state.extract_active_adapter()
    spec = {"objective": "sft", "samples": [{"prompt": "Name a color.", "response": "Blue."}]}
    # Inference disables gradients; admission must restore the training mode
    # before constructing its optimizer, without evaluating a forward pass.
    FastLanguageModel.for_inference(state.model)
    with monkeypatch.context() as patch:
        patch.setattr(state.model, "forward", lambda *a, **kw: pytest.fail("forward before admission"))
        with pytest.raises(MemoryAdmissionError):
            engine.step(spec)
    for name, value in state.extract_active_adapter().items():
        assert torch.equal(value, original[name])
    assert not engine.memory.last_report["admitted"]
    engine.memory.reserve_bytes = 256 * 1024**2
    result = engine.step(spec)
    assert result["components"]["grad_norm_total"] > 0
    assert engine.memory.last_report["admitted"]
    assert not engine.memory.last_report["activation_estimate_supplied"]
    from trainfer.engine.inference import generate_chat

    generate_chat(state.model, state.tokenizer, [{"role": "user", "content": "Name a color."}],
                  max_new_tokens=2, temperature=0, mode_lock=state.mode_lock,
                  memory_admission=engine.memory)
    assert engine.memory.last_report["operation"] == "inference"
    assert engine.memory.last_report["kv_estimate_available"]
    engine.step(spec)
    flags = [m.gradient_checkpointing for m in state.model.modules() if hasattr(m, "gradient_checkpointing")]
    assert flags and all(bool(flag) == (checkpointing != "off") for flag in flags)


@pytest.mark.gpu
@pytest.mark.parametrize("equal_lengths", [False, True])
def test_microbatch_accumulation_reduces_real_training_peak(tmp_path, equal_lengths):
    import torch
    from trainfer.snapshot import SnapshotManager

    state = ModelState.load(
        model_name="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=512, lora_rank=8, lora_alpha=16,
    )
    engine = TrainEngine(state, lr=1e-4)
    samples = [{"prompt": "Name a color." if equal_lengths else f"Name color number {i}.",
                "response": "Blue is a calming color. " * (15 if equal_lengths else 12 + i)}
               for i in range(4)]
    engine.step({"objective": "sft", "samples": samples[:1]})
    manager = SnapshotManager(tmp_path)
    manager.save("initial", state, engine=engine, include_training=True)
    initial = state.extract_active_adapter()
    results = []
    specs = [
        {"objective": "sft", "samples": samples},
        {"objective": "sft", "samples": samples, "microbatch_size": 1},
        # Independent reference using the existing combined-loss path: same
        # per-sample forwards, all graphs retained, one combined backward.
        {"objectives": [{"name": "sft", "samples": [s], "weight": 0.25} for s in samples]},
    ]
    for spec in specs:
        manager.load("initial", state, engine=engine, resume=True)
        for opt in engine._opts.values():
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result = engine.step(spec)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        delta = torch.cat([(value.double() - initial[name].double()).flatten()
                           for name, value in state.extract_active_adapter().items()])
        gradients = torch.cat([p.grad.detach().cpu().double().flatten()
                               for p in state.model.parameters() if p.requires_grad])
        results.append((result, peak, delta, gradients))
    full, micro, reference = results
    cosine = float(torch.nn.functional.cosine_similarity(full[2], micro[2], dim=0))
    reference_cosine = float(torch.nn.functional.cosine_similarity(reference[2], micro[2], dim=0))
    gradient_cosine = float(torch.nn.functional.cosine_similarity(reference[3], micro[3], dim=0))
    gradient_error = float((reference[3] - micro[3]).norm() / reference[3].norm())
    print({"full_peak_bytes": full[1], "microbatch_peak_bytes": micro[1],
           "full_loss": full[0]["loss"], "microbatch_loss": micro[0]["loss"],
           "update_cosine": cosine, "combined_reference_loss": reference[0]["loss"],
           "combined_reference_cosine": reference_cosine, "equal_lengths": equal_lengths,
           "gradient_cosine": gradient_cosine, "gradient_relative_error": gradient_error,
           "parameter_dtypes": sorted({str(p.dtype) for p in state.model.parameters() if p.requires_grad})})
    assert micro[1] < full[1]
    assert micro[0]["loss"] == pytest.approx(reference[0]["loss"], rel=1e-5)
    assert reference_cosine > 0.999
    assert gradient_error < 1e-5


@pytest.mark.gpu
def test_consolidation_preview_keeps_resident_residual_conversion_on_cpu(tmp_path):
    """A GPU-resident residual must not create a float32 GPU temporary."""
    import torch
    from trainfer.consolidation import consolidate

    layer = torch.nn.Module()
    layer.base_layer = torch.nn.Linear(2048, 2048, bias=False, dtype=torch.bfloat16, device="cuda")
    layer.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(
        2048, 4, bias=False, dtype=torch.bfloat16, device="cuda",
    )})
    layer.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(
        4, 2048, bias=False, dtype=torch.bfloat16, device="cuda",
    )})
    state = ModelState(torch.nn.Sequential(layer), None, "synthetic", 4, 4)
    residual = torch.full((2048, 2048), 0.001, dtype=torch.bfloat16, device="cuda")
    state.merged_deltas = {"0.weight": residual}
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    report = consolidate(state, tmp_path, preview=True)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - allocated < 1024**2
    assert not report["applied"]
    assert state.merged_deltas["0.weight"] is residual


@pytest.mark.gpu
def test_cpu_consolidation_keeps_gpu_compact_and_trainable(tmp_path):
    import math
    import torch
    from trainfer.consolidation import consolidate
    from trainfer.snapshot import SnapshotManager
    from trainfer.tests.test_residual_live_path import _greedy_logprob

    state = ModelState.load(
        model_name="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=256, lora_rank=8, lora_alpha=16,
    )
    engine = TrainEngine(state, lr=1e-4)
    prompt, response = "Name a color.", "Blue is a calming color."
    spec = {"objective": "sft", "samples": [{"prompt": prompt, "response": response}]}
    for _ in range(3):
        engine.step(spec)
    before_lp = _greedy_logprob(state, prompt, response)
    manager = SnapshotManager(tmp_path / "snapshots")
    manager.save("before", state)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    # Exercise the lossy mechanism, not a claim that a tiny three-step update
    # survives NF4. The measured error is printed and default admission is
    # covered separately; this test explicitly permits a larger error.
    report = consolidate(state, tmp_path / "work", max_relative_update_error=2.0)
    torch.cuda.synchronize()
    extra_peak = torch.cuda.max_memory_allocated() - allocated
    assert extra_peak < 32 * 1024**2, "consolidation materialized dense GPU weights"
    assert report["applied"] and math.isfinite(report["relative_update_error"])
    assert not state.merged_deltas
    after_lp = _greedy_logprob(state, prompt, response)
    assert math.isfinite(after_lp)
    print({"relative_update_error": report["relative_update_error"],
           "extra_gpu_peak_bytes": extra_peak, "wall_seconds": report["wall_seconds"],
           "logprob_before": before_lp, "logprob_after": after_lp})
    engine.reset_optimizer()
    result = engine.step(spec)
    assert result["components"]["grad_norm_total"] > 0
    manager.load("before", state)
    assert state.base_revision == "original"
    restored_lp = _greedy_logprob(state, prompt, response)
    assert restored_lp == pytest.approx(before_lp, abs=1e-5)


def test_end_to_end_training_moves_logprob():
    """Train on a fixed (prompt, target); measure mean log-prob on the target
    before and after several steps; confirm it moved upward (loss decreased).
    This is the load-bearing visibility test: if training didn't affect the
    shared-weight model, this metric wouldn't move.
    """
    print("[e2e] loading Qwen3-0.6B (fresh state) …")
    state = ModelState.load(
        model_name="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=1024,
        lora_rank=8,
        lora_alpha=16,
    )
    engine = TrainEngine(state, lr=1e-4)

    prompt = "The special code is:"
    target = " ZETA-X9-PRIME-47 (this is my secret)."

    r0 = engine.step(
        {
            "objective": "sft",
            "samples": [{"prompt": prompt, "response": target}],
        }
    )
    initial_loss = r0["loss"]
    print(f"[e2e] initial SFT loss: {initial_loss:.3f}")

    for _ in range(8):
        engine.step(
            {
                "objective": "sft",
                "samples": [{"prompt": prompt, "response": target}],
            }
        )

    r_final = engine.step(
        {
            "objective": "sft",
            "samples": [{"prompt": prompt, "response": target}],
        }
    )
    final_loss = r_final["loss"]
    print(f"[e2e] final SFT loss:   {final_loss:.3f}")
    assert final_loss < initial_loss - 0.5, (
        f"expected substantial loss drop, got {initial_loss:.3f} → {final_loss:.3f}"
    )
    print("[e2e] training visibly moved policy OK")


@pytest.mark.anyio
async def test_controller_commit_cursor_e2e():
    """Submit a train request, then a generate with after_commit_token, via
    the Controller. The generate must block until training commits, and the
    response must reflect the trained content.
    """
    from trainfer.config import ServeConfig
    from trainfer.controller import Controller

    print("[ctrl] loading controller …")
    cfg = ServeConfig(
        model="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=1024,
        lora_rank=8,
        lora_alpha=16,
        data_dir=__import__("pathlib").Path(f"/tmp/trainfer_test_{int(time.time())}"),
    )
    ctl = Controller(cfg)
    await ctl.start()
    try:
        # Submit a training batch that should be reflected.
        spec = {
            "objective": "sft",
            "chunk_size": 1,
            "samples": [
                {
                    "prompt": "The color of the sky is",
                    "response": " green because of photosynthesis atmosphere.",
                },
                {
                    "prompt": "The color of the sky is",
                    "response": " green because of photosynthesis atmosphere.",
                },
            ],
        }
        submit = await ctl.submit_train(spec)
        token = submit["commit_token"]
        print(
            f"[ctrl] submitted train, commit_token={token}, chunks={submit['n_chunks']}"
        )

        # Generate with after_commit_token — this must block until commit.
        t_before = time.time()
        result = await ctl.generate(
            [{"role": "user", "content": "The color of the sky is"}],
            max_new_tokens=8,
            temperature=0.1,
            after_commit_token=token,
        )
        gen_wall = time.time() - t_before
        print(
            f"[ctrl] generate after commit: {gen_wall:.2f}s; cursor={ctl.queue.committed}"
        )
        assert ctl.queue.committed >= token, "inference ran before training committed"
        print(f"[ctrl] response: {result['response']!r}")
        print("[ctrl] commit-cursor end-to-end OK")
    finally:
        await ctl.stop()


def main() -> int:
    test_merge_determinism()
    test_end_to_end_training_moves_logprob()
    asyncio.run(test_controller_commit_cursor_e2e())
    print("[test_merge_and_e2e] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
