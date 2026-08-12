"""C-3 / C-8 regression tests.

C-3: reset_optimizer() must clear the KTO z0 EMA so stale drift info
     from the pre-snapshot trajectory doesn't corrupt post-restore KTO steps.

C-8: OOM (or any exception) during backward/step must clean up stale
     gradients and free CUDA cache to prevent cascading failures.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from trainfer.engine.train import TrainEngine
from trainfer.objectives.kto import _Z0_EMA

pytestmark = pytest.mark.cpu_only


# ------------------------------------------------------------------ C-3 tests


def test_reset_optimizer_clears_z0_ema():
    """reset_optimizer() must empty the KTO _Z0_EMA dict."""
    # Seed the EMA with fake entries.
    _Z0_EMA[(12345, 0.1)] = 0.42
    _Z0_EMA[(12345, 0.5)] = 0.99
    assert len(_Z0_EMA) >= 2

    engine = TrainEngine.__new__(TrainEngine)
    engine._opts = {"": MagicMock()}
    engine.reset_optimizer()

    assert engine._opts == {}, "optimizer instances not cleared"
    assert len(_Z0_EMA) == 0, (
        "C-3: _Z0_EMA not cleared by reset_optimizer() — stale drift info "
        "will corrupt KTO steps after snapshot restore"
    )


def test_z0_ema_survives_normal_operation():
    """Sanity: _Z0_EMA should NOT be cleared by normal optimizer creation."""
    _Z0_EMA.clear()
    _Z0_EMA[(99999, 0.1)] = 0.5

    engine = TrainEngine.__new__(TrainEngine)
    engine._opts = {}
    engine.state = MagicMock()
    engine.state.model.parameters.return_value = [
        nn.Parameter(torch.randn(4, 4, requires_grad=True))
    ]
    engine.per_objective = False
    engine.per_objective_lr = {}
    engine.lr = 1e-4

    # Getting an optimizer should NOT clear the EMA.
    _opt = engine._optimizer()
    assert (99999, 0.1) in _Z0_EMA, "EMA should survive normal optimizer creation"

    # Clean up.
    _Z0_EMA.clear()


# ------------------------------------------------------------------ C-8 tests


class _FakeModel(nn.Module):
    """Minimal model with one trainable parameter for OOM testing."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, 4))


def _make_engine_for_oom() -> tuple[TrainEngine, _FakeModel]:
    """Create a minimal TrainEngine with a real model and optimizer."""
    model = _FakeModel()
    state = MagicMock()
    state.model = model
    state.model.parameters = model.parameters
    state.merged_deltas = {}

    engine = TrainEngine.__new__(TrainEngine)
    engine.state = state
    engine.lr = 1e-4
    engine.grad_clip = 1.0
    engine.per_objective = False
    engine.per_objective_lr = {}
    engine._opts = {}
    engine.default_watchlist = []
    return engine, model


def test_oom_during_backward_cleans_up_gradients():
    """When backward() raises, zero_grad(set_to_none=True) must fire."""
    engine, model = _make_engine_for_oom()

    # Pre-set a gradient to simulate stale state.
    model.w.grad = torch.ones_like(model.w)

    # Create a loss that will raise during backward.
    loss = MagicMock()
    loss.backward = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

    # Patch get_objective to return a function that produces our rigged loss.
    def fake_objective(model, tokenizer, samples, **kwargs):
        return {"loss": loss, "components": {}}

    # Hide bitsandbytes so the engine uses plain AdamW (works on CPU).
    with (
        patch("trainfer.engine.train.get_objective", return_value=fake_objective),
        patch.dict("sys.modules", {"bitsandbytes": None}),
    ):
        # Clear cached optimizer so it rebuilds without bnb.
        engine._opts.clear()
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            engine.step(
                {
                    "objective": "sft",
                    "samples": [{"prompt": "hi", "response": "hello"}],
                }
            )

    # The optimizer should have been created and zero_grad called.
    # Verify gradients were cleaned up (set_to_none=True sets grad to None).
    assert model.w.grad is None, "C-8: stale gradients not cleaned up after OOM"


def test_oom_during_step_cleans_up():
    """When opt.step() raises, gradients must still be cleaned up."""
    engine, model = _make_engine_for_oom()

    # Create a real loss that can backward successfully.
    x = model.w.sum()
    loss_val = x * 2.0  # simple differentiable loss

    def fake_objective(model_arg, tokenizer, samples, **kwargs):
        return {"loss": loss_val, "components": {}}

    # Hide bnb, then break step.
    with (
        patch("trainfer.engine.train.get_objective", return_value=fake_objective),
        patch.dict("sys.modules", {"bitsandbytes": None}),
    ):
        engine._opts.clear()

        def broken_step(self_opt, *args, **kwargs):
            raise RuntimeError("CUDA out of memory during step")

        with patch.object(torch.optim.AdamW, "step", broken_step):
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                engine.step(
                    {
                        "objective": "sft",
                        "samples": [{"prompt": "hi", "response": "hello"}],
                    }
                )

    # Gradients should be cleaned up.
    assert model.w.grad is None, "C-8: gradients not cleaned after step() OOM"


def test_oom_does_not_corrupt_next_step():
    """After an OOM, the next successful step must work normally."""
    engine, model = _make_engine_for_oom()

    call_count = [0]

    def fake_objective(model_arg, tokenizer, samples, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: produce a loss that OOMs during backward.
            loss = MagicMock()
            loss.backward = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
            return {"loss": loss, "components": {}}
        else:
            # Second call: normal loss.
            return {"loss": model_arg.w.sum(), "components": {}}

    with (
        patch("trainfer.engine.train.get_objective", return_value=fake_objective),
        patch.dict("sys.modules", {"bitsandbytes": None}),
    ):
        engine._opts.clear()

        # First step: OOM.
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            engine.step(
                {
                    "objective": "sft",
                    "samples": [{"prompt": "hi", "response": "hello"}],
                }
            )

        # Second step: should succeed without cascading failures.
        result = engine.step(
            {
                "objective": "sft",
                "samples": [{"prompt": "hi", "response": "hello"}],
            }
        )

    assert result["loss"] is not None, "step after OOM should succeed"
    assert not result.get("skipped", False), "step after OOM should not be skipped"
