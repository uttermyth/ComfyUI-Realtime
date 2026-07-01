import asyncio
import threading
import time

import pytest

from comfyui_realtime.engine.executor_bridge import bridge_sync_iterator


async def test_yields_items_from_worker_thread():
    def factory():
        yield 1
        yield 2
        yield 3

    stop_event = threading.Event()
    results = [item async for item in bridge_sync_iterator(factory, stop_event)]
    assert results == [1, 2, 3]


async def test_runs_on_a_different_thread_than_the_caller():
    caller_thread_id = threading.get_ident()
    worker_thread_ids = []

    def factory():
        worker_thread_ids.append(threading.get_ident())
        yield "x"

    stop_event = threading.Event()
    async for _ in bridge_sync_iterator(factory, stop_event):
        pass
    assert worker_thread_ids[0] != caller_thread_id


async def test_propagates_exceptions_from_the_worker():
    def factory():
        yield "ok"
        raise RuntimeError("boom")

    stop_event = threading.Event()
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in bridge_sync_iterator(factory, stop_event):
            pass


async def test_abandoning_iteration_sets_stop_event_promptly():
    stop_event = threading.Event()
    release_worker = threading.Event()

    def factory():
        for i in range(1000):
            if stop_event.is_set():
                return
            yield i
        release_worker.set()

    async def consume_two_then_abandon():
        count = 0
        async for _ in bridge_sync_iterator(factory, stop_event):
            count += 1
            if count == 2:
                break

    await consume_two_then_abandon()
    # Give the worker thread a moment to observe the flag and exit its loop.
    await asyncio.sleep(0.2)
    assert stop_event.is_set()
    assert not release_worker.is_set()  # worker stopped well before iteration 1000
