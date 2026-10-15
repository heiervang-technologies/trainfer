"""Admission rejects known shortfalls without attempting model computation."""

from types import SimpleNamespace

import pytest
import torch

from trainfer.memory import MemoryAdmission
from trainfer.errors import MemoryAdmissionError
from trainfer.state import ModelState

pytestmark = pytest.mark.cpu_only


def capacity(monkeypatch, available):
    monkeypatch.setattr("trainfer.memory.cuda_capacity", lambda device: {
        "device": str(device), "available_bytes": available,
    })


def test_training_counts_missing_gradients_lazy_optimizer_and_workspace(monkeypatch):
    model = torch.nn.Linear(4, 3, bias=False)
    opt = torch.optim.AdamW(model.parameters())
    capacity(monkeypatch, 10000)
    guard = MemoryAdmission(reserve_bytes=32, training_workspace_bytes=100)
    report = guard.training(model, opt)
    assert report["gradient_bytes"] == 48
    assert report["new_optimizer_bytes"] == 100
    assert report["required_bytes"] == 248
    model(torch.ones(1, 4)).sum().backward()
    opt.step()
    report = guard.training(model, opt)
    assert report["gradient_bytes"] == 0
    assert report["new_optimizer_bytes"] == 0
    opt.zero_grad(set_to_none=True)
    assert guard.training(model, opt)["required_bytes"] == 148


def test_reserve_rejection_reports_shortfall_without_mutating_weights(monkeypatch):
    model = torch.nn.Linear(4, 3, bias=False)
    original = model.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters())
    capacity(monkeypatch, 179)
    guard = MemoryAdmission(reserve_bytes=32)
    with pytest.raises(MemoryAdmissionError, match="180|148"):
        guard.training(model, opt)
    assert not guard.last_report["admitted"]
    assert torch.equal(original, model.weight)
    assert not opt.state


def test_residual_merge_budget_includes_existing_and_new_layers(monkeypatch):
    from trainfer.tests.test_adapter_restart import Adapter

    state = ModelState(torch.nn.Sequential(Adapter(), Adapter()), None, "fake", 2, 2)
    state.model[0].lora_B["default"].weight.data.zero_()
    state.merged_deltas = {"0.weight": torch.ones(3, 4, dtype=torch.bfloat16)}
    capacity(monkeypatch, 335)
    guard = MemoryAdmission(reserve_bytes=0)
    with pytest.raises(MemoryAdmissionError):
        guard.residual_merge(state)
    assert guard.last_report["required_bytes"] == 2 * 12 * 2 + 12 * 24
    assert state.merges_applied == 0


def test_unknown_optimizer_is_explicitly_not_estimated(monkeypatch):
    model = torch.nn.Linear(4, 3)
    capacity(monkeypatch, 10000)
    report = MemoryAdmission(reserve_bytes=0).training(model, torch.optim.SGD(model.parameters(), lr=0.1))
    assert not report["optimizer_estimate_available"]
    assert not report["activation_estimate_supplied"]


def test_cpu_execution_does_not_query_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: pytest.fail("queried CUDA for CPU model"))
    model = torch.nn.Linear(4, 3)
    assert MemoryAdmission().training(model, torch.optim.AdamW(model.parameters())) is None


def test_cached_memory_is_counted_without_assuming_total_device_capacity(monkeypatch):
    from trainfer.memory import cuda_capacity

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (100, 1000))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 300)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 350)
    assert cuda_capacity("cuda:0")["available_bytes"] == 150


def test_inference_kv_estimate_respects_gqa_and_prompt_plus_generation(monkeypatch):
    model = torch.nn.Linear(4, 3, dtype=torch.bfloat16)
    model.config = SimpleNamespace(num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, hidden_size=16)
    capacity(monkeypatch, 1000000)
    report = MemoryAdmission(reserve_bytes=0, inference_workspace_bytes=100).inference(
        model, torch.ones(1, 10, dtype=torch.long), 20,
    )
    assert report["kv_bytes"] == 2 * 3 * 2 * 4 * 2 * (10 + 20 + 256)
    assert report["required_bytes"] == report["kv_bytes"] + 100


@pytest.mark.parametrize("stream", [False, True])
def test_inference_rejects_before_generation_and_releases_lock(monkeypatch, stream):
    import sys
    from trainfer.engine.inference import generate_chat, generate_chat_stream
    from trainfer.scheduling import ModelScheduler

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            return {"input_ids": torch.ones(1, 5, dtype=torch.long)}

    model = torch.nn.Linear(4, 3)
    model.generate = lambda **kwargs: pytest.fail("generation ran despite rejection")
    capacity(monkeypatch, 0)
    # Streaming imports transformers lazily; rejection must precede streamer
    # construction, so this test does not require transformers installed.
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        TextIteratorStreamer=lambda *a, **kw: pytest.fail("streamer constructed before admission"),
    ))
    lock = ModelScheduler()
    guard = MemoryAdmission(reserve_bytes=1)
    with pytest.raises(MemoryAdmissionError):
        if stream:
            list(generate_chat_stream(model, Tokenizer(), [], mode_lock=lock, memory_admission=guard))
        else:
            generate_chat(model, Tokenizer(), [], mode_lock=lock, memory_admission=guard)
    assert lock.acquire(timeout=0.1)
    lock.release()


def test_loading_controls_reach_backend_and_resume_metadata(monkeypatch):
    import sys
    from trainfer.tests.test_adapter_restart import Adapter

    calls = {}
    model = Adapter()

    def pretrained(**kwargs):
        calls["base"] = kwargs
        return model, None

    def peft(model, **kwargs):
        calls["adapter"] = kwargs
        return model

    monkeypatch.setattr("trainfer.state.install", lambda: None)
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=SimpleNamespace(
        from_pretrained=pretrained, get_peft_model=peft,
    )))
    state = ModelState.load("fake", lora_rank=4, lora_alpha=8, lora_dropout=0.1,
                            lora_target_modules=["q_proj", "v_proj"],
                            gradient_checkpointing="off", model_dtype="float32")
    assert calls["base"]["dtype"] is torch.float32
    assert calls["adapter"]["target_modules"] == ["q_proj", "v_proj"]
    assert calls["adapter"]["use_gradient_checkpointing"] is False
    assert state.training_config["lora_dropout"] == 0.1
    with pytest.raises(ValueError, match="nonempty"):
        ModelState.load("fake", lora_target_modules=[])
    model.extra = torch.nn.Linear(4, 3)
    with pytest.raises(ValueError, match="non-LoRA trainable"):
        ModelState.load("fake")
