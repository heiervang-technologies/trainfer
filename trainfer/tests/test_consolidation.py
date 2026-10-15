"""CPU bitsandbytes tests for preview, rejection, rollback, and packed restore."""

import pytest
import torch

from trainfer.consolidation import consolidate, error_metrics
from trainfer.quantized import pack
from trainfer.snapshot import SnapshotManager
from trainfer.state import ModelState

bnb = pytest.importorskip("bitsandbytes")

pytestmark = pytest.mark.cpu_only


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = bnb.nn.Linear4bit(
            16, 16, bias=False, quant_type="nf4", compress_statistics=True,
        ).to("cpu")
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(16, 4, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 16, bias=False)})
        self.scaling = {"default": 1.0}


def make_state():
    torch.manual_seed(27)
    dense = Projection()
    dense.base_layer = torch.nn.Linear(16, 16, bias=False)
    return ModelState(torch.nn.Sequential(Projection(), dense), None, "synthetic", 4, 4)


def assert_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_metrics_detect_erased_update_and_activation_sensitivity():
    base = torch.ones(2, 2)
    update = torch.tensor([[0.01, 0.0], [0.0, 0.1]])
    metrics = error_metrics(base, update, base, torch.tensor([[10.0, 0.0]]))
    assert metrics["relative_update_error"] == pytest.approx(1.0)
    assert metrics["retained_norm_ratio"] == 0.0
    assert metrics["update_cosine"] is None
    assert metrics["activation_error_squared_norm"] == pytest.approx(0.01)


def test_preview_and_error_gate_leave_model_unchanged(tmp_path):
    state = make_state()
    before = pack(state.model[0].base_layer)
    adapter = state.extract_active_adapter()
    rng = torch.get_rng_state()
    report = consolidate(state, tmp_path, preview=True)
    assert report["relative_update_error"] > 0
    assert not report["applied"]
    assert_equal(before, pack(state.model[0].base_layer))
    assert_equal(adapter, state.extract_active_adapter())
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="exceeds limit"):
        consolidate(state, tmp_path, max_relative_update_error=0)
    assert_equal(before, pack(state.model[0].base_layer))
    assert_equal(adapter, state.extract_active_adapter())
    assert not list(tmp_path.iterdir())


def test_cpu_workspace_rejected_before_dense_allocation(tmp_path):
    state = make_state()
    with pytest.raises(ValueError, match="CPU workspace"):
        consolidate(state, tmp_path, max_cpu_workspace_bytes=1)
    assert state.base_revision == "original"


def test_default_gate_rejects_update_lost_below_quantization_resolution(tmp_path):
    state = ModelState(torch.nn.Sequential(Projection()), None, "synthetic", 4, 4)
    with torch.no_grad():
        state.model[0].lora_A["default"].weight.fill_(1)
        state.model[0].lora_B["default"].weight.fill_(1e-10)
    original = pack(state.model[0].base_layer)
    report = consolidate(state, tmp_path, preview=True)
    assert not report["default_error_gate_passes"]
    with pytest.raises(ValueError, match="not smaller than the update"):
        consolidate(state, tmp_path)
    assert_equal(original, pack(state.model[0].base_layer))
    assert state.base_revision == "original"


def test_consolidation_is_compact_and_snapshots_restore_both_revisions(tmp_path):
    state = make_state()
    manager = SnapshotManager(tmp_path / "snapshots")
    original = pack(state.model[0].base_layer)
    manager.save("before", state)
    parameter = state.model[0].base_layer.weight
    report = consolidate(state, tmp_path / "work")
    assert report["applied"]
    assert state.model[0].base_layer.weight is parameter
    assert not state.merged_deltas
    assert not hasattr(parameter, "_residual_delta")
    assert state.model[0].lora_B["default"].weight.count_nonzero() == 0
    assert state.model[0].lora_A["default"].weight.count_nonzero() > 0
    consolidated = pack(state.model[0].base_layer)
    revision = state.base_revision
    manager.save("after", state)
    # Zero update must not quantize the base again or change its revision.
    second = consolidate(state, tmp_path / "work")
    assert not second["applied"]
    assert state.base_revision == revision
    manager.load("before", state)
    assert_equal(original, pack(state.model[0].base_layer))
    assert state.base_revision == "original"
    manager.load("after", state)
    assert_equal(consolidated, pack(state.model[0].base_layer))
    assert state.base_revision == revision


def test_mid_apply_failure_rolls_back_every_projection(tmp_path, monkeypatch):
    from trainfer import quantized

    state = make_state()
    before = [pack(module.base_layer) for module in state.model]
    adapter = state.extract_active_adapter()
    original_apply = quantized.apply
    calls = 0

    def fail_once(base, tensors):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated copy failure")
        return original_apply(base, tensors)

    monkeypatch.setattr(quantized, "apply", fail_once)
    with pytest.raises(RuntimeError, match="simulated"):
        consolidate(state, tmp_path)
    for module, weights in zip(state.model, before):
        assert_equal(weights, pack(module.base_layer))
    assert_equal(adapter, state.extract_active_adapter())
    assert state.base_revision == "original"


def test_snapshot_copy_failure_preserves_current_base_revision(tmp_path, monkeypatch):
    from trainfer import quantized

    state = make_state()
    manager = SnapshotManager(tmp_path / "snapshots")
    manager.save("before", state)
    consolidate(state, tmp_path / "work")
    before = [pack(module.base_layer) for module in state.model]
    revision = state.base_revision
    original_apply = quantized.apply
    calls = 0

    def fail_once(base, tensors):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated restore failure")
        return original_apply(base, tensors)

    monkeypatch.setattr(quantized, "apply", fail_once)
    with pytest.raises(RuntimeError, match="simulated"):
        manager.load("before", state)
    for module, weights in zip(state.model, before):
        assert_equal(weights, pack(module.base_layer))
    assert state.base_revision == revision
