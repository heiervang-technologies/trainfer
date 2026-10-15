"""Reentrant model lock with bounded inference preference.

Scheduling happens between complete model operations, never during backward or
generation. Inference may overtake queued updates up to a configured burst;
then the oldest update gets a turn. Update order remains FIFO.
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time


@dataclass(eq=False)
class _Waiter:
    kind: str


class ModelScheduler:
    def __init__(self, inference_burst: int = 4):
        if inference_burst < 1:
            raise ValueError("inference_burst must be positive")
        self.inference_burst = inference_burst
        self._condition = threading.Condition()
        self._waiting: deque[_Waiter] = deque()
        self._owner: int | None = None
        self._depth = 0
        self._burst = 0
        self._kind = "inference"
        self._started = 0.0
        self._counts = {"inference": 0, "update": 0}
        self._wait_seconds = {"inference": 0.0, "update": 0.0}
        self._hold_seconds = {"inference": 0.0, "update": 0.0}

    def _next(self):
        inference = next((w for w in self._waiting if w.kind == "inference"), None)
        update = next((w for w in self._waiting if w.kind == "update"), None)
        if update is not None and (inference is None or self._burst >= self.inference_burst):
            return update
        return inference

    def _acquire(self, kind, blocking=True, timeout=-1):
        if kind not in self._counts:
            raise ValueError(f"unknown operation kind {kind!r}")
        if not blocking and timeout != -1:
            raise ValueError("cannot specify timeout for non-blocking acquisition")
        if timeout < 0 and timeout != -1:
            raise ValueError("timeout must be nonnegative or -1")
        started = time.monotonic()
        owner = threading.get_ident()
        with self._condition:
            if self._owner == owner:
                self._depth += 1
                return True
            waiter = _Waiter(kind)
            self._waiting.append(waiter)
            self._condition.notify_all()
            def ready():
                return self._owner is None and self._next() is waiter
            acquired = ready() if not blocking else self._condition.wait_for(
                ready, timeout=None if timeout == -1 else timeout,
            )
            self._waiting.remove(waiter)
            if not acquired:
                self._condition.notify_all()
                return False
            self._owner, self._depth, self._kind = owner, 1, kind
            self._started = time.monotonic()
            self._counts[kind] += 1
            self._wait_seconds[kind] += self._started - started
            self._burst = self._burst + 1 if kind == "inference" else 0
            return True

    def acquire(self, blocking=True, timeout=-1):
        return self._acquire("inference", blocking, timeout)

    def release(self):
        with self._condition:
            if self._owner != threading.get_ident():
                raise RuntimeError("cannot release a model operation owned by another thread")
            self._depth -= 1
            if not self._depth:
                self._hold_seconds[self._kind] += time.monotonic() - self._started
                self._owner = None
                self._condition.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    @contextmanager
    def operation(self, kind):
        self._acquire(kind)
        try:
            yield self
        finally:
            self.release()

    def stats(self):
        with self._condition:
            return {
                "inference_burst": self.inference_burst,
                "waiting": {k: sum(w.kind == k for w in self._waiting) for k in self._counts},
                "operations": dict(self._counts),
                "wait_seconds": dict(self._wait_seconds),
                "hold_seconds": dict(self._hold_seconds),
            }


def model_operation(lock, kind):
    """Accept plain reentrant locks for external/fake ModelState callers."""
    return lock.operation(kind) if isinstance(lock, ModelScheduler) else lock
