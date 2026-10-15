"""Failed dependencies must never reach either generation path."""

import asyncio

import pytest

from trainfer.controller import Controller
from trainfer.errors import NotFoundError, TaskFailedError, TimeoutError
from trainfer.queue import ComputeQueue

pytestmark = pytest.mark.cpu_only


def test_unknown_dependency_prevents_generation():
    async def scenario():
        c = Controller.__new__(Controller)
        c._shutting_down = False
        c.queue = ComputeQueue()
        # Deliberately no model: reaching generation would raise AttributeError.
        with pytest.raises(NotFoundError):
            await c.generate([], after_commit_token=9)
        with pytest.raises(NotFoundError):
            await anext(c.stream_generate([], after_commit_token=9))

    asyncio.run(scenario())


def test_timeout_and_completed_failure_are_not_success():
    async def scenario():
        c = Controller.__new__(Controller)
        c._shutting_down = False
        c.queue = ComputeQueue()
        task = await c.queue.submit("train", {})
        with pytest.raises(TimeoutError):
            await c.wait_for_commit(task.token, timeout=0.001)

        async def fail(task):
            raise RuntimeError("failed update")

        await c.queue.start(fail)
        await task.done.wait()
        with pytest.raises(TaskFailedError):
            await c.generate([], after_commit_token=task.token)
        with pytest.raises(TaskFailedError):
            await anext(c.stream_generate([], after_commit_token=task.token))
        await c.queue.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_http_rejects_unknown_dependency_before_success_headers(tmp_path, stream):
    from fastapi.testclient import TestClient
    from trainfer.config import ServeConfig
    from trainfer.server import create_app

    # No lifespan/model load: dependency rejection must happen before generation.
    app = create_app(ServeConfig(data_dir=tmp_path))
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "after_commit_token": 4,
        "stream": stream,
    })
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_long_merge_waits_for_completion_without_detaching_result():
    async def scenario():
        c = Controller.__new__(Controller)
        c._shutting_down = False
        c.queue = ComputeQueue()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def merge(task):
            assert task.kind == "merge"
            entered.set()
            await release.wait()
            return {"preview": True, "wall_seconds": 332.4}

        original_wait = c.queue.wait_for
        deadlines = []

        async def record_wait(task, timeout=None):
            deadlines.append(timeout)
            return await original_wait(task, timeout=timeout)

        c.queue.wait_for = record_wait
        await c.queue.start(merge)
        request = asyncio.create_task(c.request_merge(strategy="cpu_requantize", preview=True))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            assert not request.done()
            assert deadlines == [None], "accepted long merge must not lose its result at a fixed deadline"
            release.set()
            response = await asyncio.wait_for(request, timeout=1)
            assert response["result"]["wall_seconds"] == 332.4
            assert response["commit_token"] == c.queue.committed
        finally:
            release.set()
            await asyncio.gather(request, return_exceptions=True)
            await c.queue.stop()

    asyncio.run(scenario())
