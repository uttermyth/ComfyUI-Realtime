"""VLLMProvider loads a local HuggingFace transformers-format model
directory (same on-disk shape as TransformersLLMProvider -- config.json +
.safetensors weights + tokenizer files) via vLLM's synchronous LLMEngine,
running fully in-process (EngineCoreClient's InprocClient, selected
automatically because LLMEngine constructs its client with
asyncio_mode=False) -- no EngineCore subprocess, no multiprocessing.spawn,
no re-exec of ComfyUI's own main.py.

Like LlamaCppLLMProvider and TransformersLLMProvider, this provider now
holds a threading.Lock -- one generate() call in flight per provider
instance at a time. This is a deliberate regression from this provider's
previous AsyncLLMEngine-based design, which held no lock and relied on
AsyncLLMEngine's native continuous batching across concurrently-submitted
requests from different realtime sessions sharing one provider instance
(see registry.py's PipelineRegistry). That design hit an unresolved
production bug: AsyncLLMEngine forces its EngineCore onto a subprocess via
multiprocessing.spawn once CUDA is already initialized in the parent
process (as ComfyUI's other nodes do before this provider loads), and that
subprocess was observed to die silently in production with no traceback,
no OOM-killer log entry, and no GPU Xid fault -- only surfacing later as
EngineDeadError on the next request. Moving to sync LLMEngine's in-process
InprocClient eliminates that entire bug class at the root, at the cost of
serializing concurrent sessions against one provider instance -- see
docs/superpowers/specs/2026-07-08-vllm-provider-redesign-design.md for the
full design rationale, including why this regression was accepted and
scoped as a cross-provider (not vLLM-specific) concern to revisit later.

Unlike LlamaCppLLMProvider/TransformersLLMProvider, this provider's lock is
acquired via `await asyncio.to_thread(self._lock.acquire)`, not a raw
blocking `self._lock.acquire()` call. This is a deliberate, traced fix, not
an inconsistency: a raw blocking acquire() directly inside an async
function freezes the whole single-threaded event loop while contended,
which would prevent whichever call currently holds the lock (suspended on
its own bridge_sync_iterator queue) from ever being resumed to finish and
release it -- a real deadlock under genuine concurrent sessions on one
event loop, not just serialization. See
docs/superpowers/specs/2026-07-08-vllm-provider-redesign-design.md's
Architecture and Out of Scope sections for the full trace and its
cross-provider blast radius (the same raw-acquire() pattern exists in
LlamaCppLLMProvider, TransformersLLMProvider, piper_tts.py, and
pocket_tts.py -- not fixed here, flagged as separate follow-up work).

Single-GPU only: this provider assumes tensor_parallel_size=1. Multi-GPU
via the engine_args escape hatch is unverified after this redesign -- vLLM's
tensor-parallel worker processes are launched through a mechanism
independent of the EngineCoreClient choice this redesign changes, and
whether that path reintroduces the spawn/re-exec problem has not been
checked.

vLLM auto-detects quant_method from the checkpoint's own config.json at
engine-construction time. The `quantization` constructor parameter is only
an explicit override for the rare case a caller needs to force a specific
method; leaving it empty lets vLLM's own detection run.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
import logging
import threading
import uuid
import asyncio
from typing import AsyncIterator, Iterator

import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams

try:
    # vllm >= 0.22 moved tokenizer loading out of transformers_utils and into
    # its own top-level package.
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer

from ..engine.executor_bridge import bridge_sync_iterator
from .base import ChatMessage, GenerationDelta, GenerationOptions

logger = logging.getLogger("comfyui_realtime")


class VLLMProvider:
    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
        quantization: str = "",
        enforce_eager: bool = False,
        trust_remote_code: bool = False,
        engine_args: str = "",
        system_prompt: str = "",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "VLLMProvider requires a CUDA GPU (torch.cuda.is_available() is False)."
                "use TransformersLLMProvider or LlamaCppLLMProvider on non-CUDA hardware."
            )

        self._tokenizer = get_tokenizer(model_path, trust_remote_code=trust_remote_code)
        if self._tokenizer.chat_template is None:
            raise ValueError(
                f"Model at {model_path!r} has no tokenizer chat_template. "
                "VLLMProvider only supports instruct/chat-tuned models "
                "with a chat template"
            )

        engine_args_obj = EngineArgs(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
        )
        if max_model_len is not None:
            engine_args_obj.max_model_len = max_model_len
        if dtype != "auto":
            engine_args_obj.dtype = dtype
        if quantization:
            engine_args_obj.quantization = quantization
        engine_args_obj = self._apply_engine_args_overlay(engine_args_obj, engine_args)

        # Sync LLMEngine constructs its EngineCoreClient with
        # asyncio_mode=False, which resolves to InprocClient -- EngineCore
        # runs in this same process, with no subprocess, no
        # multiprocessing.spawn, no re-exec of ComfyUI's own main.py. The
        # AsyncLLMEngine-based version of this provider needed to
        # temporarily clear main_module.__file__ around engine construction
        # to survive that spawn; that workaround is gone because the
        # subprocess it worked around no longer exists. See this module's
        # docstring and
        # docs/superpowers/specs/2026-07-08-vllm-provider-redesign-design.md
        # for the full rationale.
        self._engine = LLMEngine.from_engine_args(engine_args_obj)
        self._lock = threading.Lock()
        self._system_prompt = system_prompt

    @staticmethod
    def _apply_engine_args_overlay(engine_args_obj, engine_args_json: str):
        """Overlays a user-supplied JSON object onto engine_args_obj via
        dataclasses.replace"""
        if not engine_args_json:
            return engine_args_obj
        try:
            overrides = json.loads(engine_args_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"engine_args is not valid JSON: {exc}") from exc
        if not isinstance(overrides, dict):
            raise ValueError(
                f"engine_args must be a JSON object, got {type(overrides).__name__}"
            )
        valid_fields = {f.name for f in dataclasses.fields(type(engine_args_obj))}
        for key in overrides:
            if key not in valid_fields:
                suggestion = difflib.get_close_matches(key, valid_fields, n=1)
                hint = f" did you mean {suggestion[0]!r}?" if suggestion else ""
                raise ValueError(f"unknown engine_args key {key!r}.{hint}")
        return dataclasses.replace(engine_args_obj, **overrides)

    def unload(self) -> None:
        # Best-effort LLMEngine shutdown before dropping the reference.
        # Unlike the old AsyncLLMEngine-based version, this engine no
        # longer runs a background asyncio loop or an out-of-process
        # EngineCore -- but the registry still calls unload() to reclaim a
        # loaded model's VRAM when a pipeline is replaced/unregistered (see
        # registry.py's _unload_orphaned_providers_locked), so a real
        # shutdown attempt still matters here. vLLM's shutdown API has
        # moved across versions, so this tries known method names
        # defensively rather than assuming one exact name; it's a
        # deliberate no-op if the installed version exposes neither.
        if self._engine is not None:
            shutdown = getattr(self._engine, "shutdown", None)
            if shutdown is None:
                shutdown = getattr(self._engine, "shutdown_background_loop", None)
            if shutdown is not None:
                shutdown()
        self._engine = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _drive_to_completion(
        self,
        request_id: str,
        prompt_text: str,
        sampling_params: SamplingParams,
        stop_event: threading.Event,
    ) -> Iterator[str]:
        """Runs on bridge_sync_iterator's worker thread: adds one request to
        the (single-flight, lock-protected) engine and drives it to
        completion via a blocking step() loop, yielding incremental text
        deltas. This is the seam a future redesign recovering cross-session
        concurrency would replace -- see the design spec's Architecture
        section -- without needing to change generate()'s lock/bridge/
        GenerationDelta contract."""
        logger.debug(
            "vllm request %s: add_request, prompt length %d chars",
            request_id, len(prompt_text),
        )
        generated_text = ""
        step_count = 0
        try:
            self._engine.add_request(request_id, prompt_text, sampling_params)
            while self._engine.has_unfinished_requests():
                if stop_event.is_set():
                    logger.warning(
                        "vllm request %s: cancelled after %d steps, aborting",
                        request_id, step_count,
                    )
                    self._engine.abort_request(request_id)
                    return
                for output in self._engine.step():
                    if output.request_id != request_id:
                        continue  # defensive; single-flight lock means this shouldn't fire
                    step_count += 1
                    candidate_text = output.outputs[0].text
                    delta = candidate_text.removeprefix(generated_text)
                    generated_text = candidate_text
                    if delta:
                        yield delta
                    if output.finished:
                        logger.debug(
                            "vllm request %s: finished after %d steps",
                            request_id, step_count,
                        )
                        return
        except Exception:
            # Never swallow the traceback here -- bridge_sync_iterator's
            # worker() will also forward this exception to the caller via
            # its own queue-based mechanism, but that path only needs to
            # carry the exception object, not print it anywhere. Logging it
            # here, at the point it actually escapes this method, is what
            # guarantees a real trail exists in this codebase's own logs if
            # a hard crash follows shortly after (same posture as the
            # already-committed d8cf5c7 fix for websocket_handler.py's
            # task-done callback, which stopped swallowing tracebacks via
            # %r instead of exc_info=).
            logger.error(
                "vllm request %s: unhandled exception after %d steps",
                request_id, step_count, exc_info=True,
            )
            raise

    async def generate(
        self, messages: list[ChatMessage], options: GenerationOptions
    ) -> AsyncIterator[GenerationDelta]:
        chat_messages = self._build_chat_messages(messages)
        prompt_text = self._tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        sampling_params = SamplingParams(
            temperature=options.temperature, max_tokens=options.max_tokens or 512
        )
        request_id = uuid.uuid4().hex
        stop_event = threading.Event()

        def factory():
            yield from self._drive_to_completion(
                request_id, prompt_text, sampling_params, stop_event
            )

        # Acquired via asyncio.to_thread rather than a raw blocking
        # self._lock.acquire() -- see this module's docstring for why: a
        # raw blocking acquire() here would freeze the entire event loop
        # while contended, which would prevent the lock's current holder
        # (another generate() call, suspended on its own
        # bridge_sync_iterator queue) from ever being resumed to finish and
        # release it -- a real deadlock under genuine concurrent sessions,
        # not just serialization.
        await asyncio.to_thread(self._lock.acquire)
        bridge = bridge_sync_iterator(factory, stop_event)
        try:
            async for text in bridge:
                yield GenerationDelta(text=text, finished=False)
            yield GenerationDelta(text="", finished=True)
        finally:
            # Explicitly await the inner generator's own aclose() so its
            # worker-thread join (bridge_sync_iterator's finally) completes
            # before we release the lock -- see LlamaCppLLMProvider's
            # module docstring for why relying on the async-for's implicit
            # teardown order is not sufficient here.
            await bridge.aclose()
            self._lock.release()

    def _build_chat_messages(self, messages: list[ChatMessage]) -> list[dict]:
        result = []
        if self._system_prompt:
            result.append({"role": "system", "content": self._system_prompt})
        result.extend({"role": m.role, "content": m.content} for m in messages)
        return result
