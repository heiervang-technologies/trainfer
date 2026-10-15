"""Measure the serving pause of real compact consolidation under queued load."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import time

import pytest
import unsloth  # noqa: F401
import torch

from trainfer.config import ServeConfig
from trainfer.controller import Controller

pytestmark = pytest.mark.gpu


def test_compact_consolidation_with_queued_chats_and_update(tmp_path, monkeypatch):
    asyncio.run(run_consolidation(tmp_path, monkeypatch))


async def run_consolidation(tmp_path, monkeypatch):
    from trainfer import consolidation

    ctl = Controller(ServeConfig(
        model="unsloth/qwen3-0.6b-unsloth-bnb-4bit", max_seq_length=256,
        lora_rank=8, lora_alpha=16, data_dir=tmp_path / "data",
        autoload_on_boot=False, autosave_on_exit=False,
    ))
    await ctl.start()
    original = consolidation.consolidate
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    measurement = {}

    def measured(*args, **kwargs):
        # Controller has acquired the model lock before entering this function.
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        measurement["started"] = time.perf_counter()
        loop.call_soon_threadsafe(entered.set)
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        measurement["finished"] = time.perf_counter()
        measurement["extra_gpu_bytes"] = torch.cuda.max_memory_allocated() - allocated
        return result

    monkeypatch.setattr(consolidation, "consolidate", measured)
    spec = {"objective": "sft", "samples": [
        {"prompt": "Name a color.", "response": "Blue is a calming color."},
    ]}
    try:
        for _ in range(3):
            submitted = await ctl.submit_train(spec)
            await ctl.wait_for_commit(submitted["commit_token"])
        # Explicit permissive error bound exercises application, not quality.
        merge = asyncio.create_task(ctl.request_merge(strategy="cpu_requantize", max_relative_update_error=10.0))
        await asyncio.wait_for(entered.wait(), timeout=30)
        submitted_at = time.perf_counter()
        update = await ctl.submit_train(spec)
        chats = []

        async def chat(index):
            first = None
            async for event in ctl.stream_generate(
                [{"role": "user", "content": "Name a color."}],
                max_new_tokens=8, temperature=0, parse_reasoning=False,
                after_commit_token=update["commit_token"] if index == 2 else None,
            ):
                first = first or time.perf_counter()
                if event.get("final"):
                    cursor = event["commit_cursor"]
            assert first >= measurement["finished"]
            if index == 2:
                assert cursor >= update["commit_token"]
            chats.append({"first_stream_event_seconds": first - submitted_at,
                          "latency_seconds": time.perf_counter() - submitted_at,
                          "after_update": index == 2, "cursor": cursor})

        async def wait_update():
            result = await ctl.wait_for_commit(update["commit_token"], timeout=180)
            measurement["update_completion_seconds"] = time.perf_counter() - submitted_at
            assert result.result["components"]["grad_norm_total"] > 0

        results = await asyncio.wait_for(asyncio.gather(merge, wait_update(), *(chat(i) for i in range(3))), timeout=180)
        report = results[0]["result"]
        assert report["applied"] and not ctl.state.merged_deltas
        assert ctl.queue.committed == update["commit_token"]
        assert measurement["extra_gpu_bytes"] < 32 * 1024**2
        summary = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parents[2], text=True).strip(),
                   "consolidation": {k: v for k, v in report.items() if k != "layers"},
                   "layers": len(report["layers"]), "chats": chats,
                   "merge_extra_gpu_bytes": measurement["extra_gpu_bytes"],
                   "update_completion_seconds": measurement["update_completion_seconds"],
                   "wall_seconds": time.perf_counter() - measurement["started"],
                   "notes": ["Synthetic shared-GPU integration check; no quality claim.",
                             "Explicit relative-error allowance 10 exercises application.",
                             "First stream event includes text buffering; not token-level TTFT."]}
        destination = Path(os.environ.get("TRAINFER_CONSOLIDATION_REPORT", str(tmp_path / "consolidation-serving.json")))
        destination.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        await ctl.stop()
