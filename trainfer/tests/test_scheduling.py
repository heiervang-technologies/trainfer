"""Exercise ordering under actual thread contention, without CUDA."""

import threading

import pytest

from trainfer.scheduling import ModelScheduler

pytestmark = pytest.mark.cpu_only


def test_bounded_inference_preference_preserves_update_fifo():
    scheduler = ModelScheduler(inference_burst=2)
    order = []
    threads = []

    def work(kind, name):
        with scheduler.operation(kind):
            # Production nests locks in controller, engine, and model helpers.
            with scheduler:
                order.append(name)

    with scheduler.operation("update"):
        jobs = [("update", "u0"), ("update", "u1")]
        jobs += [("inference", f"i{i}") for i in range(5)]
        for index, (kind, name) in enumerate(jobs):
            thread = threading.Thread(target=work, args=(kind, name), daemon=True)
            thread.start()
            threads.append(thread)
            with scheduler._condition:
                assert scheduler._condition.wait_for(
                    lambda: len(scheduler._waiting) == index + 1, timeout=2,
                )
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert order == ["i0", "i1", "u0", "i2", "i3", "u1", "i4"]
    stats = scheduler.stats()
    assert stats["waiting"] == {"inference": 0, "update": 0}
    assert stats["operations"] == {"inference": 5, "update": 3}


def test_timeout_removes_waiter_and_does_not_steal_ownership():
    scheduler = ModelScheduler()
    results = []

    def contender():
        results.append(scheduler.acquire(timeout=0.01))
        results.append(scheduler.acquire(blocking=False))
        with pytest.raises(RuntimeError):
            scheduler.release()

    with scheduler:
        thread = threading.Thread(target=contender)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert scheduler.stats()["waiting"]["inference"] == 0
    assert results == [False, False]
    with scheduler.operation("update"):
        pass
