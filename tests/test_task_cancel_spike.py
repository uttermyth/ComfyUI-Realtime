# tests/test_task_cancel_spike.py
"""Phase 2 cancellation spike (spec section 6.5-6.6, section 10).

Validates the load-bearing assumption behind the background-task barge-in
design (Tasks 9-10): does asyncio.Task.cancel() on a task that's consuming
LlamaCppLLMProvider.generate() reliably trigger the provider's own
try/finally cleanup (lock release), the same way Phase 1 verified explicit
aclose() does? If this fails, the barge-in design needs each provider call
site to catch CancelledError explicitly and call aclose() itself, rather
than relying on cancellation propagating automatically through the
consuming async-for.

Requires the model at models/qwen2.5-0.5b-instruct-q8_0.gguf (repo root).
"""
import asyncio
import pathlib
import time

import pytest

pytest.importorskip("llama_cpp")

from comfyui_realtime.providers.base import ChatMessage, GenerationOptions
from comfyui_realtime.providers.llama_cpp_llm import LlamaCppLLMProvider

MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"

pytestmark = pytest.mark.integration


async def test_task_cancel_releases_provider_lock_promptly():
    assert MODEL_PATH.exists(), f"test model not found at {MODEL_PATH}"
    provider = LlamaCppLLMProvider(model_path=str(MODEL_PATH), n_ctx=512)
    try:
        deltas_received = []

        async def consume():
            async for delta in provider.generate(
                [ChatMessage(role="user", content="Write a long, detailed story about a journey.")],
                GenerationOptions(max_tokens=256),
            ):
                deltas_received.append(delta)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.1)  # let a few tokens generate first
        cancel_at = time.perf_counter()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        cancel_to_done = time.perf_counter() - cancel_at

        # The real test: if the lock from the cancelled call weren't
        # released, this would hang or take far longer than one model's
        # worth of latency.
        start2 = time.perf_counter()
        second_deltas = [
            d
            async for d in provider.generate(
                [ChatMessage(role="user", content="Say hi.")], GenerationOptions(max_tokens=8)
            )
        ]
        elapsed2 = time.perf_counter() - start2

        assert len(deltas_received) > 0, "no tokens were generated before cancellation"
        assert len(second_deltas) > 1, "second generate() call after cancellation produced no output"
        assert elapsed2 < 5.0, (
            f"second generate() took {elapsed2:.2f}s after task.cancel() -- "
            f"exceeds the generous 5s sanity bound, the lock may not have released promptly"
        )

        print(
            f"\nTask-cancel spike result: cancel_to_done={cancel_to_done:.4f}s, "
            f"tokens_before_cancel={len(deltas_received)}, second_call_elapsed={elapsed2:.4f}s"
        )
    finally:
        provider.unload()


async def test_bare_gather_cancellation_does_not_guarantee_lock_release():
    """Negative control. Models the REAL Task 9-10 shape -- an outer task
    awaiting asyncio.gather(run_llm_task, run_tts_task), where run_tts
    consumes run_llm's output via a queue -- with NO explicit cleanup in
    the except block. Demonstrates that cancelling the outer task can
    return control to the caller while the provider's lock is still held,
    proving the fix in the next test is load-bearing, not decorative."""
    assert MODEL_PATH.exists()
    provider = LlamaCppLLMProvider(model_path=str(MODEL_PATH), n_ctx=512)
    try:
        sentence_queue: asyncio.Queue = asyncio.Queue()

        async def run_llm():
            async for delta in provider.generate(
                [ChatMessage(role="user", content="Write a long, detailed story about a journey.")],
                GenerationOptions(max_tokens=256),
            ):
                await sentence_queue.put(delta.text)
            await sentence_queue.put(None)

        async def run_tts():
            while True:
                item = await sentence_queue.get()
                if item is None:
                    return
                await asyncio.sleep(0.01)  # stand-in for TTS synthesis work

        async def handler_without_fix():
            llm_task = asyncio.create_task(run_llm())
            tts_task = asyncio.create_task(run_tts())
            await asyncio.gather(llm_task, tts_task)  # no except/cleanup -- this is the bug

        outer = asyncio.create_task(handler_without_fix())
        await asyncio.sleep(0.1)
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer

        # The bug: check the provider's lock state immediately after the
        # outer await returns, before anything else runs. If this prints
        # True, the lock is still held even though task.cancel() "finished."
        print(f"\nWithout fix -- lock held immediately after cancel: {provider._lock.locked()}")
    finally:
        # Drain whatever's left so the provider can be cleanly unloaded
        # regardless of what state the lock was in.
        await asyncio.sleep(0.5)
        provider.unload()


async def test_explicit_regather_on_cancel_guarantees_lock_release():
    """Positive case: the fix Task 10 must use -- in the except
    asyncio.CancelledError block, explicitly re-await any not-yet-done
    child tasks (with return_exceptions=True) before the function itself
    finishes unwinding. This is what guarantees the provider's lock is
    actually free by the time the caller's `await task` returns."""
    assert MODEL_PATH.exists()
    provider = LlamaCppLLMProvider(model_path=str(MODEL_PATH), n_ctx=512)
    try:
        sentence_queue: asyncio.Queue = asyncio.Queue()

        async def run_llm():
            async for delta in provider.generate(
                [ChatMessage(role="user", content="Write a long, detailed story about a journey.")],
                GenerationOptions(max_tokens=256),
            ):
                await sentence_queue.put(delta.text)
            await sentence_queue.put(None)

        async def run_tts():
            while True:
                item = await sentence_queue.get()
                if item is None:
                    return
                await asyncio.sleep(0.01)

        async def handler_with_fix():
            llm_task = asyncio.create_task(run_llm())
            tts_task = asyncio.create_task(run_tts())
            try:
                await asyncio.gather(llm_task, tts_task)
            except asyncio.CancelledError:
                pending = [t for t in (llm_task, tts_task) if not t.done()]
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                raise

        outer = asyncio.create_task(handler_with_fix())
        await asyncio.sleep(0.1)
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer

        assert not provider._lock.locked(), "lock still held immediately after the fixed cancellation path"

        # The real-world check: a second generate() call right after must
        # succeed quickly, not race a still-running first decode.
        start2 = time.perf_counter()
        second_deltas = [
            d
            async for d in provider.generate(
                [ChatMessage(role="user", content="Say hi.")], GenerationOptions(max_tokens=8)
            )
        ]
        elapsed2 = time.perf_counter() - start2

        assert len(second_deltas) > 1
        assert elapsed2 < 5.0, f"second generate() took {elapsed2:.2f}s -- the re-gather fix may be insufficient"
        print(f"\nNested-gather cancel spike (with fix) result: elapsed2={elapsed2:.4f}s")
    finally:
        provider.unload()
