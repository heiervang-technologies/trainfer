"""Queue-level graceful drain — pins the contract for #11.

Before this change, ``ComputeQueue.stop()`` set ``_stop`` and awaited the
worker; any tasks still in ``self._q`` were silently abandoned and their
``done`` event was never fired — so every ``wait_for(token)`` call for an
abandoned task would block until its own timeout.

``graceful_drain(deadline_s)`` fixes that:

- closes the queue to new submits (``submit`` raises ``ShuttingDownError``),
- lets the worker finish what it has pulled AND continue pulling while the
  budget holds,
- on deadline expiry stops pulling (but does not cancel the in-flight task
  mid-GPU-step — tearing the LoRA would violate invariants),
- resolves every still-pending task with ``ShutdownDroppedError`` and fires
  its ``done`` event so every waiter unblocks.

Run with: ``uv run pytest trainfer/tests/test_queue_graceful_drain.py``
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.cpu_only


async def _noop_handler(task):
    return {"ok": True}


async def _slow_handler(task):
    await asyncio.sleep(task.payload["sleep_s"])
    return {"slept": task.payload["sleep_s"]}


# ---------------------------------------------------------------- drain path


def test_graceful_drain_completes_all_pending_when_budget_sufficient():
    async def main():
        from trainfer.queue import ComputeQueue

        q = ComputeQueue(max_depth=16)
        await q.start(_noop_handler)
        tasks = [await q.submit("train", {}) for _ in range(5)]
        result = await q.graceful_drain(deadline_s=5.0)
        assert result["dropped"] == 0
        assert result["timed_out"] is False
        for t in tasks:
            assert t.done.is_set()
            assert t.error is None
        assert q.committed == tasks[-1].token

    asyncio.run(main())


def test_graceful_drain_closes_queue_to_new_submits():
    async def main():
        from trainfer.errors import ShuttingDownError
        from trainfer.queue import ComputeQueue

        q = ComputeQueue(max_depth=16)
        await q.start(_noop_handler)
        drain_task = asyncio.create_task(q.graceful_drain(deadline_s=1.0))
        await asyncio.sleep(0.05)  # let _accepting flip
        with pytest.raises(ShuttingDownError):
            await q.submit("train", {})
        await drain_task

    asyncio.run(main())


# ---------------------------------------------------------------- deadline path


def test_graceful_drain_drops_unpulled_tasks_on_deadline():
    """Queue 3 tasks that each sleep 500ms; drain with 200ms deadline.
    The in-flight task (first) runs to completion; the other two are
    dropped with ShutdownDroppedError and their done events fire."""

    async def main():
        from trainfer.errors import ShutdownDroppedError
        from trainfer.queue import ComputeQueue

        q = ComputeQueue(max_depth=16)
        await q.start(_slow_handler)
        tasks = [await q.submit("train", {"sleep_s": 0.5}) for _ in range(3)]
        result = await q.graceful_drain(deadline_s=0.2)
        for t in tasks:
            assert t.done.is_set(), f"token {t.token} never resolved"
        dropped_tasks = [t for t in tasks if isinstance(t.error, ShutdownDroppedError)]
        assert len(dropped_tasks) >= 1
        assert result["dropped"] == len(dropped_tasks)
        assert result["timed_out"] is True

    asyncio.run(main())


def test_dropped_task_waiter_resolves_instead_of_hanging():
    """A client holding a commit_token gets a deterministic resolution, not
    a timeout 60s later. ``wait_for`` returns a task carrying
    ``ShutdownDroppedError``, not an ``asyncio.TimeoutError``."""

    async def main():
        from trainfer.errors import ShutdownDroppedError
        from trainfer.queue import ComputeQueue

        q = ComputeQueue(max_depth=16)
        await q.start(_slow_handler)
        tokens = [(await q.submit("train", {"sleep_s": 0.5})).token for _ in range(3)]
        last = tokens[-1]
        drain_task = asyncio.create_task(q.graceful_drain(deadline_s=0.1))
        resolved = await asyncio.wait_for(q.wait_for(last, timeout=5.0), timeout=5.0)
        assert isinstance(resolved.error, ShutdownDroppedError)
        await drain_task

    asyncio.run(main())


# ---------------------------------------------------------------- idempotence / after


def test_drain_is_idempotent():
    async def main():
        from trainfer.queue import ComputeQueue

        q = ComputeQueue(max_depth=4)
        await q.start(_noop_handler)
        r1 = await q.graceful_drain(deadline_s=1.0)
        r2 = await q.graceful_drain(deadline_s=1.0)
        assert r1["dropped"] == 0
        assert r2["dropped"] == 0

    asyncio.run(main())


def test_submit_after_drain_raises_shutting_down():
    async def main():
        from trainfer.errors import ShuttingDownError
        from trainfer.queue import ComputeQueue

        q = ComputeQueue(max_depth=4)
        await q.start(_noop_handler)
        await q.graceful_drain(deadline_s=1.0)
        with pytest.raises(ShuttingDownError):
            await q.submit("train", {})

    asyncio.run(main())


def test_deadline_does_not_report_running_thread_as_completed():
    import threading

    async def scenario():
        from trainfer.queue import ComputeQueue
        from trainfer.errors import ShutdownDroppedError

        entered, release = threading.Event(), threading.Event()
        q = ComputeQueue()

        def handler(task):
            entered.set()
            assert release.wait(2)

        await q.start(handler)
        active = await q.submit("train", {})
        queued = await q.submit("train", {})
        assert await asyncio.to_thread(entered.wait, 1)
        try:
            result = await q.graceful_drain(deadline_s=0.001, hard_stop_grace_s=0.001)
            assert result["in_flight"]
            assert not active.done.is_set()
            assert q.committed == -1
            assert isinstance((await q.wait_for(queued.token)).error, ShutdownDroppedError)
        finally:
            release.set()
            await q.stop()
        assert active.done.is_set()
        assert active.error is None
        assert q.committed == queued.token

    asyncio.run(scenario())
