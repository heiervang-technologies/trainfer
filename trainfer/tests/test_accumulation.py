"""Compare actual SFT reductions and optimizer state across batch partitions."""

import copy
from types import SimpleNamespace

import pytest
import torch

from trainfer.engine.train import TrainEngine
from trainfer.engine.accumulation import plan, register_reduction
from trainfer.objectives import OBJECTIVES
from trainfer.state import ModelState

pytestmark = pytest.mark.cpu_only


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[ord(c) - 96 for c in text])

    def decode(self, ids):
        return "".join(chr(i + 96) for i in ids)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(27, 8)
        self.head = torch.nn.Linear(8, 27)

    @property
    def device(self):
        return self.embed.weight.device

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


SAMPLES = [
    {"prompt": "a", "response": "bc", "weight": 0.0},
    {"prompt": "ab", "response": "defg", "weight": 0.25},
    {"prompt": "c", "response": "ijklm", "weight": 3.0, "span_prefix": "ij"},
    {"prompt": "d", "response": "no", "weight": 2.0, "span_prefix": "no"},
    {"prompt": "e", "response": "p", "weight": 7.0},
]


@pytest.mark.parametrize("name", ["sft", "weighted_sft", "combined"])
@pytest.mark.parametrize("size", [1, 2, 4])
def test_accumulation_matches_full_update_and_adam_moments(name, size):
    torch.manual_seed(3)
    initial = Model()
    engines = [TrainEngine(ModelState(copy.deepcopy(initial), Tokenizer(), "fake", 1, 1),
                           per_objective=True, lr=0.01, grad_clip=0.2) for _ in range(2)]
    if name == "combined":
        spec = {"objectives": [{"name": "sft", "weight": 0.3},
                               {"name": "weighted_sft", "weight": 1.2, "samples": SAMPLES[:3]}],
                "samples": SAMPLES}
    else:
        spec = {"objective": name, "samples": SAMPLES}
    for _ in range(2):
        expected = engines[0].step(spec)
        actual = engines[1].step({**spec, "microbatch_size": size})
        assert actual["loss"] == pytest.approx(expected["loss"], rel=1e-6)
        assert actual["components"]["grad_norm_total"] == pytest.approx(
            expected["components"]["grad_norm_total"], rel=1e-6,
        )
        for a, b in zip(engines[0].state.model.parameters(), engines[1].state.model.parameters()):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        left, right = [next(iter(e._opts.values())).state_dict() for e in engines]
        for key in left["state"]:
            for field in left["state"][key]:
                torch.testing.assert_close(left["state"][key][field], right["state"][key][field])


def test_failure_in_later_microbatch_clears_gradients_without_optimizer_step(monkeypatch):
    calls = 0

    def loss(model, tokenizer, samples, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("late forward failed")
        return {"loss": model.head.weight.square().mean()}

    monkeypatch.setitem(OBJECTIVES, "test_accumulate", loss)
    register_reduction(loss, lambda samples, kwargs: len(samples))
    model = Model()
    engine = TrainEngine(ModelState(model, Tokenizer(), "fake", 1, 1), per_objective=True)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="late forward"):
        engine.step({"objective": "test_accumulate", "samples": SAMPLES, "microbatch_size": 1})
    assert all(p.grad is None for p in model.parameters())
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert not next(iter(engine._opts.values())).state


def test_unsupported_reduction_and_sidecars_reject_before_forward(monkeypatch):
    with pytest.raises(ValueError, match="reduction contract"):
        plan({"objective": "kto", "samples": SAMPLES, "microbatch_size": 1})
    with pytest.raises(ValueError, match="sidecar"):
        plan({"objective": "sft", "samples": SAMPLES, "microbatch_size": 1,
              "batch_objectives": [{"name": "kl_anchor"}]})
    monkeypatch.setitem(OBJECTIVES, "sft", lambda *args: None)
    with pytest.raises(ValueError, match="reduction contract"):
        plan({"objective": "sft", "samples": SAMPLES, "microbatch_size": 1})


def test_controller_rejects_unsupported_accumulation_without_admission(tmp_path):
    import asyncio
    from trainfer.config import ServeConfig
    from trainfer.controller import Controller
    from trainfer.errors import InvalidInputError

    ctl = Controller(ServeConfig(data_dir=tmp_path))

    async def run():
        with pytest.raises(InvalidInputError, match="reduction contract"):
            await ctl.submit_train({"objective": "kto", "samples": SAMPLES, "microbatch_size": 1})
        assert ctl.queue.qsize() == 0

    asyncio.run(run())


@pytest.mark.parametrize("chunk_size,expected_steps", [(5, 1), (2, 3)])
def test_controller_microbatches_preserve_chunk_commit_and_step_counts(tmp_path, chunk_size, expected_steps):
    import asyncio
    from trainfer.config import ServeConfig
    from trainfer.controller import Controller

    ctl = Controller(ServeConfig(data_dir=tmp_path))
    ctl.state = ModelState(Model(), Tokenizer(), "fake", 1, 1)
    ctl.train_engine = TrainEngine(ctl.state, per_objective=True)

    async def run():
        await ctl.queue.start(ctl._handle_task)
        try:
            submitted = await ctl.submit_train({"objective": "sft", "samples": SAMPLES,
                                                "chunk_size": chunk_size, "microbatch_size": 1})
            assert submitted["n_chunks"] == expected_steps
            await ctl.wait_for_commit(submitted["commit_token"])
            opt = next(iter(ctl.train_engine._opts.values()))
            assert all(int(state["step"]) == expected_steps for state in opt.state.values())
        finally:
            await ctl.stop()

    asyncio.run(run())
