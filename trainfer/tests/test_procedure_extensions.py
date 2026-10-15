"""Non-gradient procedures retain queue, locking, failure, and resume contracts."""

import asyncio
from dataclasses import replace
import threading
import sys
from types import SimpleNamespace

import pytest
import torch

from trainfer.engine.procedures import PROCEDURES, register_procedure
from trainfer.engine.train import TrainEngine
from trainfer.snapshot import SnapshotManager
from trainfer.tests.test_snapshot_resume import make_engine

pytestmark = pytest.mark.cpu_only


class CoordinateUpdate:
    def __init__(self):
        self.calls = 0

    def run(self, engine, spec):
        # A derivative-free synthetic update verifies that the extension is
        # not restricted to returning another differentiable loss function.
        self.calls += 1
        with torch.no_grad():
            engine.state.model.lora_B["default"].weight.add_(self.calls * spec["procedure_kwargs"]["delta"])
        return {"loss": None, "components": {"calls": self.calls, "thread": threading.get_ident()}}

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        if not isinstance(state.get("calls"), int) or state["calls"] < 0:
            raise ValueError("invalid call count")
        self.calls = state["calls"]


def validate(spec):
    if spec.get("procedure_kwargs", {}).get("delta", 0) <= 0:
        raise ValueError("positive delta required")


@pytest.fixture(autouse=True)
def register():
    before = dict(PROCEDURES)
    register_procedure("test_coordinate", CoordinateUpdate, validate=validate, version="1")
    yield
    PROCEDURES.clear()
    PROCEDURES.update(before)


SPEC = {"procedure": "test_coordinate", "procedure_kwargs": {"delta": 0.01}}


def test_stateful_procedure_snapshot_reproduces_next_update(tmp_path):
    state, engine = make_engine()
    engine.step(SPEC)
    manager = SnapshotManager(tmp_path)
    manager.save("resume", state, engine=engine, include_training=True)
    expected = engine.step(SPEC)
    expected_weights = state.extract_active_adapter()
    engine.step(SPEC)
    manager.load("resume", state, engine=engine, resume=True)
    actual = engine.step(SPEC)
    assert actual == expected
    for name, value in state.extract_active_adapter().items():
        assert torch.equal(value, expected_weights[name])
    manager.load("resume", state, engine=engine)
    assert not engine._procedures  # Weights-only restore resets algorithm state.


def test_changed_procedure_version_rejects_before_weight_restore(tmp_path):
    state, engine = make_engine()
    engine.step(SPEC)
    manager = SnapshotManager(tmp_path)
    manager.save("resume", state, engine=engine, include_training=True)
    engine.step(SPEC)
    before = state.extract_active_adapter()
    PROCEDURES["test_coordinate"] = replace(PROCEDURES["test_coordinate"], version="2")
    with pytest.raises(ValueError, match="procedure version"):
        manager.load("resume", state, engine=engine, resume=True)
    for name, value in state.extract_active_adapter().items():
        assert torch.equal(value, before[name])


def test_queue_runs_one_procedure_commit_on_worker_and_validates_before_admission(tmp_path):
    from trainfer.config import ServeConfig
    from trainfer.controller import Controller
    from trainfer.errors import InvalidInputError

    ctl = Controller(ServeConfig(data_dir=tmp_path))
    ctl.state, ctl.train_engine = make_engine()
    main_thread = threading.get_ident()

    async def run():
        with pytest.raises(InvalidInputError, match="positive delta"):
            await ctl.submit_train({"procedure": "test_coordinate"})
        assert ctl.queue.qsize() == 0
        await ctl.queue.start(ctl._handle_task, on_complete=ctl._on_task_complete)
        try:
            accepted = await ctl.submit_train({**SPEC, "samples": [{}, {}, {}], "chunk_size": 1})
            assert accepted["n_chunks"] == 1
            task = await ctl.queue.wait_for(accepted["commit_token"])
            assert task.error is None
            assert task.result["components"]["thread"] != main_thread
            assert task.result["components"]["calls"] == 1
        finally:
            await ctl.stop()

    asyncio.run(run())
    assert "procedure:test_coordinate" in (tmp_path / "trajectory.jsonl").read_text()


def test_procedure_holds_model_lock_and_failure_clears_gradients():
    class Failure(CoordinateUpdate):
        def run(self, engine, spec):
            acquired = []

            def compete():
                ok = engine.state.mode_lock.acquire(timeout=0.01)
                acquired.append(ok)
                if ok:
                    engine.state.mode_lock.release()

            thread = threading.Thread(target=compete)
            thread.start()
            thread.join()
            assert acquired == [False]
            engine.state.model.lora_B["default"].weight.sum().backward()
            raise RuntimeError("procedure failed")

    register_procedure("test_failure", Failure, validate=lambda spec: None, version="1")
    state, engine = make_engine()
    with pytest.raises(RuntimeError, match="procedure failed"):
        engine.step({"procedure": "test_failure"})
    assert all(p.grad is None for p in state.model.parameters())
    assert state.mode_lock.acquire(timeout=0.1)
    state.mode_lock.release()


def test_procedure_can_compose_multiple_builtin_gradient_updates():
    from trainfer.tests.test_accumulation import Model, Tokenizer, SAMPLES
    from trainfer.state import ModelState

    class Repeat(CoordinateUpdate):
        def run(self, engine, spec):
            for _ in range(2):
                result = engine.gradient_step(spec)
            return result

    register_procedure("test_repeat", Repeat, validate=lambda spec: None, version="1")
    engine = TrainEngine(ModelState(Model(), Tokenizer(), "test", 1, 1), optimizer_class="adamw")
    engine.step({"procedure": "test_repeat", "objective": "sft", "samples": SAMPLES,
                 "microbatch_size": 1})
    assert all(int(s["step"]) == 2 for s in engine._optimizer().state.values())


def test_procedure_invalidates_inference_mode_and_surfaces_backend_failures(monkeypatch):
    state, engine = make_engine()
    state.model.cached_inference_adapter = True

    def for_training(model, **kwargs):
        del model.cached_inference_adapter

    backend = SimpleNamespace(for_training=for_training)
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=backend))
    engine.step(SPEC)
    assert not hasattr(state.model, "cached_inference_adapter")
    before = state.extract_active_adapter()

    def fail(*args, **kwargs):
        raise RuntimeError("backend mode switch failed")

    backend.for_training = fail
    with pytest.raises(RuntimeError, match="mode switch"):
        engine.step(SPEC)
    for name, value in state.extract_active_adapter().items():
        assert torch.equal(value, before[name])
