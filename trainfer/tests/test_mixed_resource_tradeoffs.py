"""Real-model integration check with reproducible mixed-workload measurements.

Synthetic engine inputs only; learning benchmarks/policies remain in cont.
Run explicitly with the GPU environment. TRAINFER_RESOURCE_REPORT optionally
persists JSON measurements outside pytest's temporary directory.
"""

import asyncio
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import pytest
import unsloth
import torch
import transformers

from trainfer.config import ServeConfig
from trainfer.controller import Controller
from trainfer.scheduling import ModelScheduler
from trainfer.snapshot import SnapshotManager

pytestmark = pytest.mark.gpu


def gpu_status():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True).strip()


def test_mixed_resource_tradeoffs(tmp_path, monkeypatch):
    assert asyncio.run(run_mixed(tmp_path, monkeypatch))


async def run_mixed(tmp_path, monkeypatch):
    cfg = ServeConfig(
        model=os.environ.get("TRAINFER_RESOURCE_MODEL", "unsloth/qwen3-0.6b-unsloth-bnb-4bit"),
        max_seq_length=512, lora_rank=8, lora_alpha=16, data_dir=tmp_path / "data",
        autosave_on_exit=False, autoload_on_boot=False,
    )
    ctl = Controller(cfg)
    await ctl.start()
    samples = [{"prompt": f"Name color number {i}.",
                "response": "Blue is a calming color. " * (4 + i)} for i in range(4)]
    spec = {"objective": "sft", "samples": samples, "chunk_size": 4, "microbatch_size": 1}
    rows = []
    try:
        warm = await ctl.submit_train(spec)
        await ctl.wait_for_commit(warm["commit_token"])
        async for _ in ctl.stream_generate([{"role": "user", "content": "Request 0: name a color."}],
                                           max_new_tokens=24, temperature=0, parse_reasoning=False):
            pass
        snapshots = SnapshotManager(tmp_path / "snapshots")
        await asyncio.to_thread(snapshots.save, "initial", ctl.state, engine=ctl.train_engine, include_training=True)

        # Record true generated-token delivery, before text buffering/decoding.
        from trainfer.engine import inference
        original_stream = inference.generate_chat_stream
        local = threading.local()
        chats = {}
        starts = {}
        switch = {}

        def stream(model, tokenizer, messages, **kwargs):
            local.request_id = int(messages[-1]["content"].split()[1].rstrip(":"))
            yield from original_stream(model, tokenizer, messages, **kwargs)

        class TimedStreamer(transformers.TextIteratorStreamer):
            def __init__(self, *args, **kwargs):
                self.request_id = local.request_id
                super().__init__(*args, **kwargs)

            def put(self, value):
                if not (self.skip_prompt and self.next_tokens_are_prompt):
                    record = chats[self.request_id]
                    now = time.perf_counter()
                    record.setdefault("first_token_seconds", now - starts[self.request_id])
                    record.setdefault("first_token_cursor", ctl.queue.committed)
                    record.setdefault("first_token_at", now)
                    record["last_token_at"] = now
                    record["tokens"] += value.numel()
                super().put(value)

        monkeypatch.setattr(inference, "generate_chat_stream", stream)
        monkeypatch.setattr(transformers, "TextIteratorStreamer", TimedStreamer)
        for method in ("for_training", "for_inference"):
            original = getattr(unsloth.FastLanguageModel, method)

            def timed(*args, _original=original, _method=method, **kwargs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = _original(*args, **kwargs)
                torch.cuda.synchronize()
                switch.setdefault(_method, []).append(time.perf_counter() - start)
                return result

            monkeypatch.setattr(unsloth.FastLanguageModel, method, staticmethod(timed))

        final_adapters = None
        settings = ["rlock", "burst1", "burst4"]
        for repeat in range(3):
            # Rotate order to distribute warm-up/drift across the policies.
            for setting in settings[repeat:] + settings[:repeat]:
                await asyncio.to_thread(snapshots.load, "initial", ctl.state, engine=ctl.train_engine, resume=True)
                ctl.state.mode_lock = threading.RLock() if setting == "rlock" else ModelScheduler(int(setting[-1]))
                chats.clear()
                starts.clear()
                switch.clear()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before_gpu = gpu_status()
                event_offset = ctl.trajectory.size()
                started = time.perf_counter()
                commits = []
                update_latencies = []

                async def update():
                    start = time.perf_counter()
                    submitted = await ctl.submit_train(spec)
                    commits.append(submitted["commit_token"])
                    await ctl.wait_for_commit(submitted["commit_token"])
                    update_latencies.append(time.perf_counter() - start)

                # All updates are admitted in their task's first loop turn.
                updates = [asyncio.create_task(update()) for _ in range(4)]
                await asyncio.sleep(0)
                assert len(commits) == 4

                async def chat(index):
                    starts[index] = time.perf_counter()
                    dependency = commits[index % len(commits)] if index % 2 == 0 else None
                    chats[index] = {"tokens": 0, "dependency": dependency}
                    async for event in ctl.stream_generate(
                        [{"role": "user", "content": f"Request {index}: name a color."}],
                        max_new_tokens=24, temperature=0, parse_reasoning=False,
                        after_commit_token=dependency,
                    ):
                        if event.get("final"):
                            chats[index]["final_cursor"] = event["commit_cursor"]
                    chats[index]["latency_seconds"] = time.perf_counter() - starts[index]

                await asyncio.wait_for(asyncio.gather(*updates, *(chat(i) for i in range(6))), timeout=180)
                torch.cuda.synchronize()
                wall = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated()
                assert ctl.queue.committed == max(commits)
                assert sorted(commits) == list(range(min(commits), max(commits) + 1))
                for record in chats.values():
                    assert record["tokens"] > 0
                    if record["dependency"] is not None:
                        assert record["first_token_cursor"] >= record["dependency"]
                events = list(ctl.trajectory.iter_events(since_offset=event_offset))
                assert sum(e["kind"] == "train_step" for e in events) == 4
                assert sum(e["kind"] == "inference" for e in events) == 6
                adapter = ctl.state.extract_active_adapter()
                if final_adapters is None:
                    final_adapters = adapter
                else:
                    # Interleaving inference cannot change this deterministic
                    # sequence of training updates or leave stale model state.
                    for name, value in adapter.items():
                        assert torch.equal(value, final_adapters[name]), name
                raw_tokens = sum(r["tokens"] for r in chats.values())
                decode_tokens = sum(max(0, r["tokens"] - 1) for r in chats.values())
                decode_seconds = sum(r["last_token_at"] - r["first_token_at"] for r in chats.values())
                rows.append({
                    "policy": setting, "repeat": repeat, "wall_seconds": wall,
                    "peak_allocated_bytes": peak, "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "generated_tokens": raw_tokens, "mixed_tokens_per_second": raw_tokens / wall,
                    "decode_tokens_per_second": decode_tokens / decode_seconds if decode_seconds else None,
                    "updates_per_second": 4 / wall, "update_latency_seconds": update_latencies,
                    "chats": [{k: v for k, v in r.items() if not k.endswith("_at")} for r in chats.values()],
                    "mode_switch_seconds": {k: {"count": len(v), "total": sum(v)} for k, v in switch.items()},
                    "scheduler": ctl.state.mode_lock.stats() if isinstance(ctl.state.mode_lock, ModelScheduler) else None,
                    "gpu_before": before_gpu, "gpu_after": gpu_status(),
                    "correctness": {"updates": 4, "chats": 6, "dependencies_met": True, "identical_final_adapters": True},
                })
                print(json.dumps(rows[-1]), flush=True)
        report = {
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=Path(__file__).resolve().parents[2], text=True).strip(),
            "model": cfg.model, "rank": cfg.lora_rank, "alpha": cfg.lora_alpha,
            "max_seq_length": cfg.max_seq_length, "microbatch_size": 1, "samples_per_update": 4,
            "max_generated_tokens": 24, "temperature": 0,
            "packages": {p: version(p) for p in ("torch", "unsloth", "transformers", "bitsandbytes")},
            "torch_threads": torch.get_num_threads(), "rows": rows,
            "notes": ["Shared GPU; no other workloads stopped.",
                      "Current engine with unprioritized RLock is the control, not original-main performance.",
                      "Mode-switch timing synchronizes CUDA for every policy.",
                      "TTFT is first generated token delivered to streamer, including queue/dependency wait.",
                      "Synthetic engine workload; no learning-quality claim."],
        }
        destination = Path(os.environ.get("TRAINFER_RESOURCE_REPORT", str(tmp_path / "mixed-resources.json")))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        return True
    finally:
        await ctl.stop()
