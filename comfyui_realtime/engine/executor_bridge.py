"""Thread-to-event-loop bridge for synchronous provider libraries.

Every bundled provider (llama-cpp-python, piper-tts, pywhispercpp) wraps a
synchronous, blocking library. This module is the one place that dispatches
such a call to a worker thread and streams its results back to the asyncio
event loop -- so no provider implementation has to hand-roll its own
thread/queue bridging. A blocking call running directly on the loop freezes
all of ComfyUI, not just one session.

Cancellation contract: the worker thread checks `stop_event` between items
and stops promptly when it's set. If the consumer abandons the `async for`
early, this generator's `finally` block sets `stop_event` itself, so a
provider's worker thread always learns to stop even if the caller never
explicitly signals it -- "provider releases at the next boundary" is
automatic for every provider built on this bridge, not something each
provider has to remember to wire up.
"""
from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator, Callable, Iterator, TypeVar

T = TypeVar("T")

_SENTINEL = object()


async def bridge_sync_iterator(
    factory: Callable[[], Iterator[T]],
    stop_event: threading.Event,
) -> AsyncIterator[T]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker() -> None:
        try:
            for item in factory():
                if stop_event.is_set():
                    return
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the consumer below, not swallowed
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop_event.set()
        await asyncio.to_thread(thread.join, timeout=5.0)
