"""Optimizer extensions preserve live parameter identity and resume semantics."""

from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest
import torch

from trainfer.engine.optimizers import OPTIMIZERS, get_optimizer, register_optimizer
from trainfer.engine.train import TrainEngine
from trainfer.snapshot import SnapshotManager
from trainfer.tests.test_snapshot_resume import make_engine, step

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def restore_registry():
    saved = dict(OPTIMIZERS)
    yield
    OPTIMIZERS.clear()
    OPTIMIZERS.update(saved)


def momentum(params, *, lr, **kwargs):
    return torch.optim.SGD(params, lr=lr, **kwargs)


def test_custom_optimizer_resume_reproduces_next_update(tmp_path):
    register_optimizer("test_momentum", momentum, version="1")
    state, _ = make_engine()
    engine = TrainEngine(state, optimizer_class="test_momentum", optimizer_kwargs={"momentum": 0.8}, lr=0.03)
    step(engine)
    manager = SnapshotManager(tmp_path)
    manager.save("saved", state, engine=engine, include_training=True)
    expected_loss = step(engine)
    expected = state.extract_active_adapter()
    step(engine)
    manager.load("saved", state, engine=engine, resume=True)
    actual_loss = step(engine)
    assert torch.equal(actual_loss, expected_loss)
    for name, value in state.extract_active_adapter().items():
        assert torch.equal(value, expected[name])


@pytest.mark.parametrize("change", ["version", "order", "kwargs"])
def test_incompatible_factory_restore_rejects_before_mutation(tmp_path, change):
    reverse = False

    def factory(params, *, lr, **kwargs):
        torch.rand(1)  # Optimizers may use random initial projections.
        return torch.optim.SGD(list(reversed(params)) if reverse else params, lr=lr, **kwargs)

    register_optimizer("test_factory", factory, version="1")
    state, _ = make_engine()
    engine = TrainEngine(state, optimizer_class="test_factory")
    step(engine)
    manager = SnapshotManager(tmp_path)
    manager.save("saved", state, engine=engine, include_training=True)
    step(engine)
    before = state.extract_active_adapter()
    if change == "version":
        OPTIMIZERS["test_factory"] = replace(OPTIMIZERS["test_factory"], version="2")
    elif change == "order":
        reverse = True
    else:
        engine.optimizer_kwargs["momentum"] = 0.8
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="version|group order|configuration"):
        manager.load("saved", state, engine=engine, resume=True)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in state.extract_active_adapter().items():
        assert torch.equal(value, before[name])


def test_factory_cannot_replace_live_parameters():
    def factory(params, *, lr):
        return torch.optim.SGD([torch.nn.Parameter(p.detach().clone()) for p in params], lr=lr)

    register_optimizer("test_copy", factory, version="1")
    state, _ = make_engine()
    engine = TrainEngine(state, optimizer_class="test_copy")
    with pytest.raises(ValueError, match="exactly the live"):
        engine._optimizer()
    assert not engine._opts


def test_custom_per_objective_instances_and_support_contract():
    register_optimizer("test_momentum", momentum, version="1")
    state, _ = make_engine()
    engine = TrainEngine(state, optimizer_class="test_momentum", per_objective=True,
                         per_objective_lr={"sft": 0.2}, lr=0.1)
    a, b = engine._optimizer("sft"), engine._optimizer("kto")
    assert a is not b
    assert a.param_groups[0]["lr"] == 0.2
    assert b.param_groups[0]["lr"] == 0.1
    register_optimizer("test_single", momentum, version="1", supports_per_objective=False)
    engine = TrainEngine(state, optimizer_class="test_single", per_objective=True)
    with pytest.raises(ValueError, match="per-objective"):
        engine._optimizer("sft")


def test_explicit_names_and_factory_failures_do_not_silently_fallback(monkeypatch):
    with pytest.raises(ValueError, match="unknown optimizer"):
        get_optimizer("typo")

    def fail(*args, **kwargs):
        raise RuntimeError("backend construction failed")

    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace(optim=SimpleNamespace(AdamW8bit=fail)))
    state, _ = make_engine()
    with pytest.raises(RuntimeError, match="backend construction"):
        TrainEngine(state)._optimizer()


def test_optimizer_options_are_portable_and_detached_from_caller_mutation():
    state, _ = make_engine()
    options = {"betas": [0.9, 0.99]}
    engine = TrainEngine(state, optimizer_class="adamw", optimizer_kwargs=options)
    options["betas"][0] = 0.1
    assert engine.optimizer_kwargs["betas"] == [0.9, 0.99]
    with pytest.raises(ValueError):
        TrainEngine(state, optimizer_kwargs={"eps": float("nan")})
    with pytest.raises(ValueError, match="cannot override"):
        TrainEngine(state, optimizer_kwargs={"lr": 0.01})


def test_custom_state_estimate_is_used_for_admission(monkeypatch):
    register_optimizer("test_momentum", momentum, version="1", estimate_state_bytes=lambda opt: 123)
    state, _ = make_engine()
    engine = TrainEngine(state, optimizer_class="test_momentum", memory_reserve_bytes=0)
    monkeypatch.setattr("trainfer.memory.cuda_capacity", lambda device: {"available_bytes": 10000})
    report = engine.memory.training(state.model, engine._optimizer())
    assert report["optimizer_estimate_available"]
    assert report["new_optimizer_bytes"] == 123


def test_extension_registration_runs_once_and_failures_surface(monkeypatch):
    from trainfer.engine import extensions

    calls = []
    entry = SimpleNamespace(name="test", value="test:register", load=lambda: lambda: calls.append("registered"))
    monkeypatch.setattr(extensions, "_LOADED", set())
    monkeypatch.setattr(extensions, "entry_points", lambda **kwargs: [entry])
    extensions.load_extensions()
    extensions.load_extensions()
    assert calls == ["registered"]
    entry.value = "test:bad"
    entry.load = lambda: None
    with pytest.raises(TypeError, match="registrar"):
        extensions.load_extensions()
