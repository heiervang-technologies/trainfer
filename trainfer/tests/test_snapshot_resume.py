"""Snapshot publication failures and next-update equivalence after resume."""

import json
import random

import numpy as np
import pytest
import torch

from trainfer.engine.train import TrainEngine
from trainfer.snapshot import SnapshotManager
from trainfer.state import ModelState
from trainfer.tests.test_adapter_restart import Adapter

pytestmark = pytest.mark.cpu_only


def make_engine():
    state = ModelState(Adapter(), None, "synthetic", 2, 2)
    state.reset_active_adapter()
    # Plain AdamW per-objective, so the test is independent of optional bnb.
    engine = TrainEngine(state, lr=0.01, per_objective=True)
    return state, engine


def step(engine):
    model = engine.state.model
    x = torch.randn(5, 4) + random.random() + float(np.random.random())
    target = torch.randn(5, 3)
    opt = engine._optimizer("sft")
    opt.zero_grad()
    loss = (model(x) - target).square().mean()
    loss.backward()
    opt.step()
    return loss.detach().clone()


def test_resume_reproduces_next_update_and_random_state(tmp_path):
    state, engine = make_engine()
    manager = SnapshotManager(tmp_path)
    step(engine)
    manager.save("resume", state, engine=engine, include_training=True)
    expected_loss = step(engine)
    expected_weights = state.extract_active_adapter()
    for _ in range(3):
        step(engine)
    manager.load("resume", state, engine=engine, resume=True)
    actual_loss = step(engine)
    assert torch.equal(actual_loss, expected_loss)
    for name, tensor in state.extract_active_adapter().items():
        assert torch.equal(tensor, expected_weights[name])


def test_weights_only_restore_drops_optimizer(tmp_path):
    state, engine = make_engine()
    manager = SnapshotManager(tmp_path)
    step(engine)
    manager.save("weights", state)
    assert engine._opts
    manager.load("weights", state, engine=engine)
    assert not engine._opts
    with pytest.raises(ValueError, match="weights only"):
        manager.load("weights", state, engine=engine, resume=True)


def test_failed_overwrite_keeps_previous_snapshot(tmp_path, monkeypatch):
    from trainfer import snapshot

    state, engine = make_engine()
    manager = SnapshotManager(tmp_path)
    step(engine)
    manager.save("same", state)
    expected = state.extract_active_adapter()
    step(engine)
    old_manifest = (tmp_path / "same" / "manifest.json").read_bytes()

    def fail(source, destination):
        raise OSError("simulated disk failure before manifest publication")

    with monkeypatch.context() as patch:
        patch.setattr(snapshot.os, "replace", fail)
        with pytest.raises(OSError, match="simulated"):
            manager.save("same", state)
    assert (tmp_path / "same" / "manifest.json").read_bytes() == old_manifest
    manager.load("same", state)
    for name, tensor in state.extract_active_adapter().items():
        assert torch.equal(tensor, expected[name])
    assert len(list((tmp_path / "same").glob("generation-*"))) == 1


def test_empty_overwrite_does_not_restore_stale_residual(tmp_path):
    from trainfer.tests.test_snapshot_residual_bind import _make_state_with_fake_model

    state = _make_state_with_fake_model()
    manager = SnapshotManager(tmp_path)
    state.merged_deltas = {"layer0.weight": torch.ones(8, 8, dtype=torch.bfloat16)}
    manager.save("same", state)
    state.merged_deltas = {}
    manager.save("same", state)
    state.merged_deltas = {"layer0.weight": torch.ones(8, 8, dtype=torch.bfloat16)}
    state._apply_residual_to_model()
    manager.load("same", state)
    assert not state.merged_deltas
    assert not hasattr(state.model.layer0.base_layer.weight, "_residual_delta")


@pytest.mark.parametrize("change", ["base_model", "lora_rank", "base_revision", "checksum"])
def test_invalid_restore_leaves_live_weights_unchanged(tmp_path, change):
    state, engine = make_engine()
    manager = SnapshotManager(tmp_path)
    manager.save("bad", state)
    step(engine)
    before = state.extract_active_adapter()
    manifest_path = tmp_path / "bad" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "checksum":
        manifest["files"]["adapter"]["sha256"] = "wrong"
    else:
        manifest[change] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        manager.load("bad", state)
    for name, tensor in state.extract_active_adapter().items():
        assert torch.equal(tensor, before[name])


def test_optimizer_config_mismatch_rejected_before_weight_change(tmp_path):
    state, engine = make_engine()
    manager = SnapshotManager(tmp_path)
    step(engine)
    manager.save("resume", state, engine=engine, include_training=True)
    step(engine)
    before = state.extract_active_adapter()
    engine.lr *= 2
    with pytest.raises(ValueError, match="configuration differs"):
        manager.load("resume", state, engine=engine, resume=True)
    for name, tensor in state.extract_active_adapter().items():
        assert torch.equal(tensor, before[name])


def test_legacy_weights_snapshot_loads_only_against_original_base(tmp_path):
    from safetensors.torch import save_file

    state, engine = make_engine()
    manager = SnapshotManager(tmp_path)
    saved = state.extract_active_adapter()
    directory = tmp_path / "legacy"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({
        "base_model": state.base_model_name, "lora_rank": state.lora_rank,
        "lora_alpha": state.lora_alpha, "merges_applied": 0,
    }))
    save_file(saved, str(directory / "active_adapter.safetensors"))
    step(engine)
    manager.load("legacy", state)
    for name, tensor in state.extract_active_adapter().items():
        assert torch.equal(tensor, saved[name])
    state.base_revision = "changed"
    with pytest.raises(ValueError, match="base revision differs"):
        manager.load("legacy", state)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "..", ".", "", "x/y"])
def test_snapshot_name_is_local(tmp_path, name):
    manager = SnapshotManager(tmp_path)
    state, _ = make_engine()
    with pytest.raises(ValueError):
        manager.save(name, state)
