"""LlamaCppLLMProvider wraps llama-cpp-python (pinned ==0.3.30). Every
call is dispatched through engine/executor_bridge.py -- this provider never
runs llama-cpp-python's blocking decode on the caller's event loop.

The per-provider lock ("one inference at a time per loaded model") is held
by THIS async generator, not by the inner worker-thread generator. That matters for cancellation: an explicit `await gen.aclose()`
on this async generator throws GeneratorExit in at the suspended `yield`
below, which is *inside* the `async for` over `bridge_sync_iterator(...)`.

A subtlety the brief's first draft missed and this implementation corrects:
merely wrapping that `async for` in `try/finally: self._lock.release()` is
NOT enough to make the lock release wait for the worker thread to actually
stop. When `aclose()` propagates GeneratorExit through an `async for`, the
outer generator's own `finally` runs synchronously as part of `aclose()`
returning, but the *inner* async generator's `finally` (here,
bridge_sync_iterator's `stop_event.set(); await
asyncio.to_thread(thread.join, ...)`) is only scheduled, not awaited inline
-- it completes on a later event-loop tick, after `aclose()` has already
returned to the caller. Verified empirically: a minimal two-layer
async-generator repro shows the outer `finally` printing and `aclose()`
returning *before* the inner generator's own `await`-containing `finally`
block even starts. Concretely here, that gap let `self._lock.release()` run
while generation #1's worker thread was still mid-decode, so a second
`generate()` call could acquire the lock and start a second worker thread
calling `Llama.decode()` concurrently with the first -- llama.cpp has no
defense against concurrent decode calls on one context and aborts the
process (GGML_ASSERT failures in the KV cache / allocator) when this
happens.

The fix: hold an explicit reference to the `bridge_sync_iterator(...)`
async generator and `await` its own `aclose()` in THIS generator's
`finally`, before releasing the lock. Awaiting a generator's `aclose()`
directly (rather than relying on `async for`'s implicit teardown order)
does run that generator to completion -- including its internal awaits --
before returning. That guarantees the worker thread has been joined (or
hit its 5s timeout) by the time the lock is released, so a second
generate() right after an explicitly closed one never races the first
call's worker thread for the underlying Llama context. Merely letting the
generator go out of scope (e.g. `break`-ing out of an `async for` without
calling aclose()) does not give even this delayed guarantee promptly --
implicit abandonment defers cleanup to a later event-loop tick (asyncio's
asyncgen finalization hook). Any caller that cancels an in-flight generate()
early MUST call `.aclose()` explicitly rather than just breaking out of its
consuming loop, to get the lock release this design depends on.
"""
from __future__ import annotations

import threading
from typing import AsyncIterator

from llama_cpp import Llama

from ..engine.executor_bridge import bridge_sync_iterator
from .base import ChatMessage, GenerationDelta, GenerationOptions


class LlamaCppLLMProvider:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        system_prompt: str = "",
    ) -> None:
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            # Gemma 4 (and other hybrid SWA/global architectures) use a ring-buffer
            # KV cache for SWA layers sized to sliding_window (e.g. 512 tokens).
            # The ring-buffer wrap-around in llama.cpp 0.3.30 has a memory-safety
            # bug when key_length == sliding_window (as in Gemma 4), causing a
            # segfault in llama_decode once total tokens exceed the window.
            # swa_full=True allocates the full n_ctx cache for SWA layers instead,
            # avoiding the rotation without changing the attention mask semantics.
            swa_full=True,
            verbose=False,
        )
        self._system_prompt = system_prompt
        self._lock = threading.Lock()

    async def generate(
        self, messages: list[ChatMessage], options: GenerationOptions
    ) -> AsyncIterator[GenerationDelta]:
        chat_messages = self._build_chat_messages(messages)
        stop_event = threading.Event()

        def factory():
            stream = self._llm.create_chat_completion(
                messages=chat_messages,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    yield delta

        self._lock.acquire()
        bridge = bridge_sync_iterator(factory, stop_event)
        try:
            async for text in bridge:
                yield GenerationDelta(text=text, finished=False)
            yield GenerationDelta(text="", finished=True)
        finally:
            # Explicitly await the inner generator's own aclose() so its
            # worker-thread join (bridge_sync_iterator's finally) completes
            # before we release the lock -- see module docstring for why
            # relying on the async-for's implicit teardown order is not
            # sufficient here.
            await bridge.aclose()
            self._lock.release()

    def _build_chat_messages(self, messages: list[ChatMessage]) -> list[dict]:
        result = []
        if self._system_prompt:
            result.append({"role": "system", "content": self._system_prompt})
        result.extend({"role": m.role, "content": m.content} for m in messages)
        return result

    def unload(self) -> None:
        del self._llm
