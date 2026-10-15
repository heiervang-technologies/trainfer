"""Explicit large-model capacity check, using a pre-downloaded local checkpoint.

Set TRAINFER_CAPACITY_MODEL to a pinned local checkpoint directory. This test
never chooses/downloads a model or claims an absolute hardware capacity limit.
"""

import json
import os
from pathlib import Path
import subprocess
import time

import pytest
import unsloth  # noqa: F401
import torch

from trainfer.consolidation import consolidate
from trainfer.engine.inference import generate_chat
from trainfer.engine.train import TrainEngine
from trainfer.errors import MemoryAdmissionError
from trainfer.state import ModelState

pytestmark = pytest.mark.gpu


def test_large_model_train_infer_train_and_merge_admission(tmp_path):
    model_name = os.environ.get("TRAINFER_CAPACITY_MODEL")
    if not model_name:
        pytest.skip("set TRAINFER_CAPACITY_MODEL to an explicitly selected local checkpoint")
    assert Path(model_name).is_dir(), "capacity test requires a pre-downloaded pinned checkpoint"
    gpu_before = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits",
    ], text=True).strip()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    state = ModelState.load(model_name, max_seq_length=1024, lora_rank=8, lora_alpha=16)
    assert {p.device.type for p in state.model.parameters()} == {"cuda"}, "capacity check requires GPU-resident weights"
    torch.cuda.synchronize()
    report = {"checkpoint": model_name, "revision": Path(model_name).name,
              "gpu_before": gpu_before, "load_seconds": time.perf_counter() - started,
              "load_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "loaded_allocated_bytes": torch.cuda.memory_allocated(),
              "trainable_parameters": sum(p.numel() for p in state.model.parameters() if p.requires_grad),
              "rank": 8, "alpha": 16, "max_seq_length": 1024,
              "samples_per_update": 4, "microbatch_size": 1, "phases": []}
    engine = TrainEngine(state, lr=1e-4, memory_reserve_bytes=1024**3,
                         training_workspace_bytes=1024**3, inference_workspace_bytes=256 * 1024**2)
    samples = [{"prompt": f"Name color number {i}.",
                "response": "Blue is a calming color. " * (60 + i)} for i in range(4)]
    spec = {"objective": "sft", "samples": samples, "microbatch_size": 1}
    # Record exact token lengths used by this engine objective.
    from trainfer.objectives._utils import build_chat_inputs
    report["sequence_lengths"] = [len(build_chat_inputs(state.tokenizer, s["prompt"], s["response"])["input_ids"])
                                  for s in samples]
    assert max(report["sequence_lengths"]) <= 1024
    for phase in ("train", "infer", "train"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if phase == "train":
            result = engine.step(spec)
            assert result["components"]["grad_norm_total"] > 0
            assert torch.isfinite(torch.tensor(result["loss"]))
            details = {"loss": result["loss"], "gradient_norm": result["components"]["grad_norm_total"]}
        else:
            text = generate_chat(state.model, state.tokenizer, [{"role": "user", "content": "Name a color."}],
                                 max_new_tokens=8, temperature=0, mode_lock=state.mode_lock,
                                 memory_admission=engine.memory)
            assert isinstance(text, str)
            details = {"response": text}
        torch.cuda.synchronize()
        record = {"phase": phase, "seconds": time.perf_counter() - started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "admission": engine.memory.last_report, **details}
        report["phases"].append(record)
        print(json.dumps(record), flush=True)
    before = state.extract_active_adapter()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with pytest.raises(MemoryAdmissionError):
        engine.memory.residual_merge(state)
    assert not state.merged_deltas and state.merges_applied == 0
    for name, value in state.extract_active_adapter().items():
        assert torch.equal(value, before[name])
    report["dense_merge_admission"] = engine.memory.last_report
    report["dense_merge_check_extra_gpu_bytes"] = torch.cuda.max_memory_allocated() - allocated
    assert report["dense_merge_check_extra_gpu_bytes"] < 32 * 1024**2
    if os.environ.get("TRAINFER_CAPACITY_PREVIEW") == "1":
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        preview = consolidate(state, tmp_path / "preview", preview=True)
        torch.cuda.synchronize()
        report["compact_preview"] = {k: v for k, v in preview.items() if k != "layers"}
        report["compact_preview"]["layers"] = len(preview["layers"])
        report["compact_preview"]["extra_gpu_bytes"] = torch.cuda.max_memory_allocated() - allocated
        assert not preview["applied"]
        assert report["compact_preview"]["extra_gpu_bytes"] < 32 * 1024**2
        for name, value in state.extract_active_adapter().items():
            assert torch.equal(value, before[name])
    report["gpu_after"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits",
    ], text=True).strip()
    report["notes"] = ["Shared GPU; other workloads unchanged.",
                       "Successful selected configuration, not absolute largest-model or context capacity.",
                       "Activation workspace budget is explicit; admission does not guarantee fit."]
    output = Path(os.environ.get("TRAINFER_CAPACITY_REPORT", str(tmp_path / "capacity.json")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
