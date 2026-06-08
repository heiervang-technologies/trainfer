"""C-1 / C-6 regression: snapshot load must bind residual onto the live model.

C-1: SnapshotManager.load() was loading merged_deltas from disk but NOT calling
     _apply_residual_to_model(). After restore, inference used base weights only.

C-6: Loading a snapshot with empty residual over a model that had a non-empty
     residual left stale _residual_delta attrs and forward hooks active.

These tests use a synthetic model with LoRA-like structure (no GPU, no Unsloth)
to verify the snapshot load path actually binds/clears the residual.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lile.snapshot import SnapshotManager
from lile.state import ModelState

pytestmark = pytest.mark.cpu_only


class FakeBaseLayer(nn.Module):
    """Mimics a PEFT base_layer with a weight Parameter."""

    def __init__(self, out_f: int, in_f: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class FakeLoraLayer(nn.Module):
    """Mimics a PEFT LoRA layer with lora_A/lora_B dicts and a base_layer."""

    def __init__(self, out_f: int, in_f: int, rank: int = 4):
        super().__init__()
        self.base_layer = FakeBaseLayer(out_f, in_f)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_f, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, out_f, bias=False)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x)


class FakeModel(nn.Module):
    """Minimal model with LoRA-like structure for testing residual binding."""

    def __init__(self):
        super().__init__()
        self.layer0 = FakeLoraLayer(8, 8)
        self.layer1 = FakeLoraLayer(8, 8)


def _make_state_with_fake_model() -> ModelState:
    """Create a ModelState with a fake LoRA model (CPU only)."""
    state = ModelState.__new__(ModelState)
    state.model = FakeModel()
    state.tokenizer = MagicMock()
    state.base_model_name = "test/fake"
    state.lora_rank = 4
    state.lora_alpha = 8
    state.merges_applied = 0
    state.merged_deltas = {}
    state._residual_hook_handles = {}
    state.frozen_ref = None
    return state


# ------------------------------------------------------------------ C-1 tests


def test_snapshot_load_applies_residual_to_model():
    """After snapshot load with non-empty residual, _residual_delta must be
    set on the base_layer weights."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mgr = SnapshotManager(root / "snaps")
        state = _make_state_with_fake_model()

        # Create a residual and save a snapshot.
        state.merged_deltas = {
            "layer0.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            "layer1.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        }
        state.merges_applied = 1
        mgr.save("with_residual", state)

        # Wipe the state to simulate a fresh model.
        state.merged_deltas = {}
        state.merges_applied = 0

        # Verify no residual is bound before load.
        assert not hasattr(state.model.layer0.base_layer.weight, "_residual_delta")
        assert not hasattr(state.model.layer1.base_layer.weight, "_residual_delta")

        # Load the snapshot — this must call _apply_residual_to_model().
        mgr.load("with_residual", state)

        # Verify residual is now bound on the live model.
        assert hasattr(state.model.layer0.base_layer.weight, "_residual_delta"), (
            "C-1: _residual_delta not bound on layer0 after snapshot load"
        )
        assert hasattr(state.model.layer1.base_layer.weight, "_residual_delta"), (
            "C-1: _residual_delta not bound on layer1 after snapshot load"
        )
        assert state.merges_applied == 1


def test_snapshot_load_registers_forward_hooks():
    """After snapshot load with non-empty residual, forward hooks must be
    registered on base_layers."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mgr = SnapshotManager(root / "snaps")
        state = _make_state_with_fake_model()

        state.merged_deltas = {
            "layer0.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        }
        state.merges_applied = 1
        mgr.save("hooked", state)

        # Reset.
        state.merged_deltas = {}
        state._residual_hook_handles = {}

        mgr.load("hooked", state)

        # Forward hooks must be registered.
        assert len(state._residual_hook_handles) > 0, (
            "C-1: no forward hooks registered after snapshot load with residual"
        )


def test_snapshot_load_residual_affects_forward():
    """The loaded residual must actually change the forward pass output
    (via the forward hook on base_layer)."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mgr = SnapshotManager(root / "snaps")
        state = _make_state_with_fake_model()

        # Capture baseline forward.
        x = torch.randn(1, 8)
        with torch.no_grad():
            base_out = state.model.layer0.base_layer(x).clone()

        # Save snapshot with a large residual.
        delta = torch.ones(8, 8, dtype=torch.bfloat16) * 10.0
        state.merged_deltas = {"layer0.weight": delta}
        state.merges_applied = 1
        mgr.save("big_residual", state)

        # Reset and reload.
        state.merged_deltas = {}
        state._residual_hook_handles = {}
        mgr.load("big_residual", state)

        # Forward should now include the residual contribution.
        with torch.no_grad():
            loaded_out = state.model.layer0.base_layer(x)

        # The output should be different from base (residual adds F.linear(x, delta)).
        diff = (loaded_out - base_out).abs().sum().item()
        assert diff > 1.0, (
            f"C-1: residual did not affect forward pass (diff={diff:.4f})"
        )


# ------------------------------------------------------------------ C-6 tests


def test_snapshot_load_empty_clears_stale_hooks():
    """Loading a snapshot with no residual over a model that had one must
    remove all _residual_delta attrs and forward hooks."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mgr = SnapshotManager(root / "snaps")
        state = _make_state_with_fake_model()

        # First: load a snapshot WITH residual (to set up hooks).
        state.merged_deltas = {
            "layer0.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        }
        state.merges_applied = 1
        mgr.save("with_res", state)

        state.merged_deltas = {}
        state._residual_hook_handles = {}
        mgr.load("with_res", state)

        # Verify hooks are present.
        assert hasattr(state.model.layer0.base_layer.weight, "_residual_delta")
        assert len(state._residual_hook_handles) > 0

        # Now save a snapshot WITHOUT residual.
        state.merged_deltas = {}
        state.merges_applied = 0
        mgr.save("no_res", state)

        # Re-apply the residual hooks so we're in the "stale hooks" state.
        state.merged_deltas = {
            "layer0.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        }
        state._apply_residual_to_model()
        assert hasattr(state.model.layer0.base_layer.weight, "_residual_delta")

        # Load the empty-residual snapshot — must clear stale hooks.
        mgr.load("no_res", state)

        # Verify: no residual deltas, no hooks, no _residual_delta attrs.
        assert state.merged_deltas == {}
        assert len(state._residual_hook_handles) == 0, (
            "C-6: stale hook handles not cleared after loading empty-residual snapshot"
        )
        assert not hasattr(state.model.layer0.base_layer.weight, "_residual_delta"), (
            "C-6: stale _residual_delta not removed after loading empty-residual snapshot"
        )


def test_snapshot_load_empty_restores_clean_forward():
    """After loading an empty-residual snapshot, forward must return to
    baseline (no residual contribution)."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mgr = SnapshotManager(root / "snaps")
        state = _make_state_with_fake_model()

        x = torch.randn(1, 8)
        with torch.no_grad():
            base_out = state.model.layer0.base_layer(x).clone()

        # Apply a big residual.
        delta = torch.ones(8, 8, dtype=torch.bfloat16) * 10.0
        state.merged_deltas = {"layer0.weight": delta}
        state._apply_residual_to_model()

        # Verify forward changed.
        with torch.no_grad():
            dirty_out = state.model.layer0.base_layer(x)
        assert (dirty_out - base_out).abs().sum().item() > 1.0

        # Save and load an empty snapshot.
        state.merged_deltas = {}
        state.merges_applied = 0
        mgr.save("clean", state)

        # Re-apply residual to simulate stale state.
        state.merged_deltas = {"layer0.weight": delta}
        state._apply_residual_to_model()

        # Load the clean snapshot.
        mgr.load("clean", state)

        # Forward must be back to baseline.
        with torch.no_grad():
            clean_out = state.model.layer0.base_layer(x)
        diff = (clean_out - base_out).abs().sum().item()
        assert diff < 1e-5, (
            f"C-6: forward not restored to baseline after clean snapshot (diff={diff:.6f})"
        )


# ------------------------------------------------------------------ round-trip


def test_snapshot_roundtrip_residual_swap():
    """Load snap A (with residual) → load snap B (no residual) → load snap A
    again. The final state must match the original."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mgr = SnapshotManager(root / "snaps")
        state = _make_state_with_fake_model()

        # Snapshot A: with residual.
        delta_a = torch.randn(8, 8, dtype=torch.bfloat16)
        state.merged_deltas = {"layer0.weight": delta_a.clone()}
        state.merges_applied = 3
        mgr.save("snap_a", state)
        fp_a = state.residual_fingerprint()

        # Snapshot B: no residual.
        state.merged_deltas = {}
        state.merges_applied = 0
        mgr.save("snap_b", state)

        # Load A → B → A.
        mgr.load("snap_a", state)
        assert state.residual_fingerprint() == fp_a
        assert state.merges_applied == 3

        mgr.load("snap_b", state)
        assert state.merged_deltas == {}
        assert state.merges_applied == 0
        assert not hasattr(state.model.layer0.base_layer.weight, "_residual_delta")

        mgr.load("snap_a", state)
        assert state.residual_fingerprint() == fp_a
        assert state.merges_applied == 3
        assert hasattr(state.model.layer0.base_layer.weight, "_residual_delta")
