"""Tests for queue admission limits and batch size caps.

Run with: pytest trainfer/tests/test_queue_admission.py
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock

from trainfer.controller import Controller
from trainfer.config import ServeConfig
from trainfer.errors import BatchTooLargeError, QueueFullError

pytestmark = pytest.mark.cpu_only


async def _scenario_batch_too_large() -> None:
    cfg = ServeConfig(max_samples_per_train_call=10)
    c = Controller(cfg)

    spec = {
        "objective": "sft",
        "samples": [{"prompt": f"p{i}", "response": f"r{i}"} for i in range(11)],
    }
    try:
        await c.submit_train(spec)
        assert False, "Should have raised BatchTooLargeError"
    except BatchTooLargeError as e:
        assert "cap is 10" in str(e)


async def _scenario_queue_full() -> None:
    cfg = ServeConfig(max_queue_depth=2)
    c = Controller(cfg)
    c.train_engine = MagicMock()

    # Manually fill the queue without starting the worker thread.
    await c.queue.try_submit("train", {"objective": "sft", "samples": [{"a": 1}]})
    await c.queue.try_submit("train", {"objective": "sft", "samples": [{"a": 2}]})

    spec = {"objective": "sft", "samples": [{"prompt": "p", "response": "r"}]}
    try:
        await c.submit_train(spec)
        assert False, "Should have raised QueueFullError"
    except QueueFullError:
        pass


async def _scenario_chunking_honors_queue_depth() -> None:
    cfg = ServeConfig(max_queue_depth=2)
    c = Controller(cfg)

    spec = {
        "objective": "sft",
        "samples": [{"a": i} for i in range(4)],
        "chunk_size": 2,
    }
    res = await c.submit_train(spec)
    assert res["n_chunks"] == 2
    assert c.queue.qsize() == 2

    spec2 = {
        "objective": "sft",
        "samples": [{"a": i} for i in range(2)],
        "chunk_size": 2,
    }
    try:
        await c.submit_train(spec2)
        assert False, "Should have raised QueueFullError"
    except QueueFullError:
        pass


async def main() -> int:
    await _scenario_batch_too_large()
    await _scenario_queue_full()
    await _scenario_chunking_honors_queue_depth()
    return 0


def test_queue_admission() -> None:
    assert asyncio.run(main()) == 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
