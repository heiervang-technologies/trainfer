"""Compute queue with commit cursor.

The invariant from LIVELEARN §3.4:

  Any inference request that arrives *after* the server returned a commit_token
  for a training request must see that training reflected in the model.

Implementation: monotonic integer cursor, single writer (the training worker),
readers block on `wait_for(token)` until the worker's completed-cursor passes
their token. This is a semaphore, not a best-effort signal, and the test in
tests/test_queue_cursor.py is written to fail under reordering.
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import (
    CommitExpiredError, TaskFailedError, ShutdownDroppedError,
    ShuttingDownError, QueueFullError,
)

log = logging.getLogger(__name__)


@dataclass
class QueueTask:
    token: int
    kind: str  # "train" | "merge" | "snapshot" | "custom"
    payload: Any
    created_at: float = field(default_factory=time.time)
    # Set by the worker when done; inference waits on this.
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: BaseException | None = None
    result: Any = None

    # Opaque batch_id lets callers group tasks for a single /v1/train call and
    # get ONE commit_token that covers all sub-batches.
    batch_id: str = ""


class ComputeQueue:
    """Single-writer async queue with a monotonic commit cursor.

    The cursor advances when a task finishes, including failures. The task's
    `error` must be checked before treating completion as a successful update;
    readers can inspect it — a failed train step still has a deterministic
    ordering with respect to subsequent inference.
    """

    def __init__(self, max_depth: int = 64, outcome_history: int = 1024) -> None:
        if max_depth < 1 or outcome_history < 1:
            raise ValueError("queue depth and outcome history must be positive")
        self._q: asyncio.Queue[QueueTask] = asyncio.Queue(maxsize=max_depth)
        self._next_token: int = 0
        self._completed_token: int = -1
        # Maps token -> completion event. Kept small by GC after wait_for.
        self._pending: dict[int, QueueTask] = {}
        # Metadata only: never retain payloads, GPU results, or tracebacks here.
        self._outcomes: collections.OrderedDict[int, QueueTask] = collections.OrderedDict()
        self._outcome_history = outcome_history
        self._batch_remaining: dict[str, int] = {}
        self._batch_failed: set[str] = set()
        self._active: QueueTask | None = None
        self._finished_tokens: set[int] = set()
        self._cursor_advanced = asyncio.Condition()
        self._worker_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        # Shutdown coordination (see ``graceful_drain``):
        #  - ``_accepting`` gates ``submit``; flipped False on drain entry.
        #  - ``_hard_stop`` tells the worker loop to exit WITHOUT pulling more
        #    tasks, even if the queue is non-empty. Set on deadline expiry.
        self._accepting: bool = True
        self._hard_stop = asyncio.Event()
        # Monotonic timestamp of the last submit(). Read by is_idle_for().
        # Initialized far in the past so the very first poll after startup
        # reads as idle; the scheduler gates on "quiet for N seconds" anyway.
        self._last_enqueue_ts: float = 0.0

    # ------------------------------------------------------------------ enqueue
    async def try_submit(
        self, kind: str, payload: Any, batch_id: str = ""
    ) -> QueueTask:
        if self._q.full():
            raise QueueFullError(
                f"compute queue depth {self.qsize()}/{self.maxsize}; retry after backoff"
            )
        return await self.submit(kind, payload, batch_id)

    async def submit_many(self, kind: str, payloads: list[Any], batch_id: str = "") -> list[QueueTask]:
        """Admit an entire batch or none; no suspension between check and enqueue."""
        if not self._accepting:
            raise ShuttingDownError("queue is draining")
        if len(payloads) > self.maxsize - self.qsize():
            raise QueueFullError("insufficient queue capacity for the entire batch")
        if batch_id and payloads:
            self._batch_remaining[batch_id] = self._batch_remaining.get(batch_id, 0) + len(payloads)
        tasks = []
        for payload in payloads:
            task = QueueTask(self._next_token, kind, payload, batch_id=batch_id)
            self._next_token += 1
            self._pending[task.token] = task
            self._q.put_nowait(task)
            tasks.append(task)
        if tasks:
            self._last_enqueue_ts = time.monotonic()
        return tasks

    async def submit(self, kind: str, payload: Any, batch_id: str = "") -> QueueTask:
        return (await self.submit_many(kind, [payload], batch_id))[0]

    # ------------------------------------------------------------------ idleness
    def is_idle_for(self, seconds: float) -> bool:
        if self._pending:
            return False
        """True iff the queue is empty AND no submit happened in the last N seconds.

        Consulted by the idle replay scheduler (§T4.1) to avoid contending with
        live train/infer work. Single-reader pattern: the scheduler polls this
        at a coarse cadence, so we don't need a lock — `asyncio.Queue.empty()`
        is safe to call from any coroutine, and the monotonic float read is
        atomic on CPython.
        """
        return (time.monotonic() - self._last_enqueue_ts) >= seconds

    # ------------------------------------------------------------------ read-side
    @property
    def maxsize(self) -> int:
        return self._q.maxsize

    def qsize(self) -> int:
        return self._q.qsize()

    @property
    def committed(self) -> int:
        return self._completed_token

    async def wait_for(self, task_or_token: int | QueueTask, timeout: float | None = None) -> QueueTask:
        """Block until the committed cursor >= token."""
        if isinstance(task_or_token, int):
            token = task_or_token
            if token < 0 or token >= self._next_token:
                raise KeyError(f"unknown token {token}")
            task = self._pending.get(token)
            if task is None:
                task = self._outcomes.get(token)
                if task is None:
                    raise CommitExpiredError(f"outcome for commit token {token} has expired")
                return task
        else:
            task = task_or_token
        if timeout is None:
            await task.done.wait()
        else:
            await asyncio.wait_for(task.done.wait(), timeout=timeout)
        return task

    def _remember_outcome(self, task: QueueTask) -> None:
        outcome = QueueTask(task.token, task.kind, None, batch_id=task.batch_id)
        if task.error is not None:
            # Preserve structured shutdown failures but strip the traceback and
            # arbitrary exception internals from the bounded historical record.
            if isinstance(task.error, ShutdownDroppedError):
                outcome.error = ShutdownDroppedError(str(task.error))
            else:
                outcome.error = TaskFailedError(f"queue task {task.token} failed")
        outcome.done.set()
        self._outcomes[task.token] = outcome
        while len(self._outcomes) > self._outcome_history:
            self._outcomes.popitem(last=False)

    def _finish(self, task: QueueTask) -> None:
        if task.batch_id in self._batch_remaining:
            if task.error is not None:
                self._batch_failed.add(task.batch_id)
            self._batch_remaining[task.batch_id] -= 1
            if self._batch_remaining[task.batch_id] == 0:
                if task.batch_id in self._batch_failed and task.error is None:
                    task.error = TaskFailedError("another task in this batch failed")
                self._batch_failed.discard(task.batch_id)
                del self._batch_remaining[task.batch_id]
        self._finished_tokens.add(task.token)
        while self._completed_token + 1 in self._finished_tokens:
            self._completed_token += 1
            self._finished_tokens.remove(self._completed_token)
        task.done.set()
        self._remember_outcome(task)
        self._pending.pop(task.token, None)

    # ------------------------------------------------------------------ worker
    async def start(
        self, handler: Callable[[QueueTask], Any],
        on_complete: Callable[[QueueTask], None] | None = None,
    ) -> None:
        """Start the single worker task. `handler` is a coroutine-or-callable."""
        if self._worker_task is not None:
            raise RuntimeError("queue already started")
        self._worker_task = asyncio.create_task(self._run(handler, on_complete))

    async def _run(self, handler, on_complete=None) -> None:
        log.info("compute queue worker started")
        while True:
            # Drain-on-stop: exit only when stop is requested AND the queue
            # has been fully processed, OR a hard-stop is signalled (deadline
            # expired — stop pulling new tasks even if some are queued).
            if self._hard_stop.is_set():
                return
            if self._stop.is_set() and self._q.empty():
                return
            try:
                task = await asyncio.wait_for(self._q.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            self._active = task
            try:
                # Synchronous model work must never block the HTTP event loop.
                result = (
                    handler(task) if inspect.iscoroutinefunction(handler)
                    else await asyncio.to_thread(handler, task)
                )
                if asyncio.iscoroutine(result):
                    result = await result
                task.result = result
            except Exception as e:
                log.exception("queue task %d (%s) failed", task.token, task.kind)
                task.error = e
            finally:
                async with self._cursor_advanced:
                    self._finish(task)
                    self._active = None
                    self._cursor_advanced.notify_all()
                    if on_complete is not None:
                        try:
                            on_complete(task)
                        except Exception:
                            log.exception("completion notification failed for task %d", task.token)

    async def stop(self) -> None:
        self._accepting = False
        self._stop.set()
        if self._worker_task is not None:
            await self._worker_task
            self._worker_task = None

    async def graceful_drain(
        self,
        deadline_s: float | None = None,
        *,
        hard_stop_grace_s: float = 30.0,
    ) -> dict[str, Any]:
        """Close the queue, drain pending, resolve remainders with ShutdownDropped.

        On entry: flip ``_accepting`` so any new ``submit`` raises
        :class:`ShuttingDownError`, then ask the worker to stop as soon as
        the queue is empty.

        While ``deadline_s`` has budget, the worker keeps pulling tasks and
        running them — the drain-on-stop semantics mean a sufficiently long
        budget completes every enqueued task naturally.

        If the budget expires with the queue non-empty, we set
        ``_hard_stop``: the worker exits after the currently-in-flight task
        returns (we never cancel mid-GPU-step — that would tear the LoRA),
        and every still-pending queue entry gets ``error =
        ShutdownDroppedError`` and ``done.set()`` so every ``wait_for``
        caller resolves deterministically.

        ``hard_stop_grace_s`` bounds how long we'll wait for the in-flight
        task after setting ``_hard_stop``. k8s operators should size this
        plus ``deadline_s`` to stay under ``terminationGracePeriodSeconds``.

        Idempotent: a second call with nothing left to do returns
        ``{"dropped": 0, "timed_out": False}``.
        """
        self._accepting = False
        self._stop.set()
        if self._worker_task is None or self._worker_task.done():
            return await self._reap_pending(timed_out=False)

        timed_out = False
        try:
            if deadline_s is not None:
                await asyncio.wait_for(
                    asyncio.shield(self._worker_task),
                    timeout=deadline_s,
                )
            else:
                await self._worker_task
        except asyncio.TimeoutError:
            timed_out = True
            # Ask the worker to stop pulling more tasks; give the in-flight
            # one a bit more time to finish so its state is consistent.
            self._hard_stop.set()
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._worker_task),
                    timeout=hard_stop_grace_s,
                )
            except asyncio.TimeoutError:
                # In-flight task is genuinely stuck; best we can do is move
                # on and let the process teardown handle it.
                pass
        if self._worker_task.done():
            self._worker_task = None
        return await self._reap_pending(timed_out=timed_out)

    async def _reap_pending(self, *, timed_out: bool) -> dict[str, Any]:
        """Drop unstarted work; never label a still-mutating task completed."""
        dropped = 0
        async with self._cursor_advanced:
            for task in list(self._pending.values()):
                if task.done.is_set() or task is self._active:
                    continue
                task.error = ShutdownDroppedError(
                    f"queue task {task.token} ({task.kind}) dropped on shutdown",
                )
                self._finish(task)
                dropped += 1
            self._cursor_advanced.notify_all()
            while not self._q.empty():
                self._q.get_nowait()
        return {"dropped": dropped, "timed_out": timed_out, "in_flight": self._active is not None}


def new_batch_id() -> str:
    return uuid.uuid4().hex[:12]
