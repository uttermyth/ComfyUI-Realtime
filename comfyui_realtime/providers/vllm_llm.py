"""VLLMProvider loads a local HuggingFace transformers-format model
directory via vLLM's synchronous LLMEngine.

BETA: real-hardware tested but several issues (a cross-provider lock deadlock risk, a cancellation race, a
multiprocessing env var gotcha, VRAM not being reclaimed on unload) identified and not yet fixed.

Getting InprocClient requires two things: asyncio_mode=False (LLMEngine's
default) AND VLLM_ENABLE_V1_MULTIPROCESSING=0 (vLLM defaults this env var to
"1", which silently forces a subprocess client regardless of asyncio_mode).
__init__ overrides this env var for just the engine-construction call and
restores it afterward.

Unlike LlamaCppLLMProvider/TransformersLLMProvider, this provider holds a
threading.Lock (one generate() call in flight per instance -- concurrent
sessions serialize) acquired via asyncio.to_thread rather than a raw
blocking acquire(), which would deadlock the event loop under real
concurrent load.

Single-GPU only (tensor_parallel_size=1); multi-GPU is unverified.
Quantization is auto-detected from the checkpoint's config.json --
`quantization` is only an override for forcing a specific method.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
import logging
import os
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

_MULTIPROCESSING_ENV_VAR = "VLLM_ENABLE_V1_MULTIPROCESSING"


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
        # Set first so __del__ never hits AttributeError on a partially
        # constructed instance (e.g. construction raising below before these
        # are normally assigned).
        self._engine = None
        self._tokenizer = None

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

        # Force VLLM_ENABLE_V1_MULTIPROCESSING to "0" for just this call so
        # LLMEngine actually resolves to InprocClient -- see module
        # docstring. multiprocess_mode is only read at construction time, so
        # restoring the prior value right after is safe.
        original_value = os.environ.get(_MULTIPROCESSING_ENV_VAR)
        if original_value not in (None, "0"):
            logger.warning(
                "Overriding %s=%r to '0' for this engine's construction -- "
                "VLLMProvider requires vLLM's EngineCore to run in-process "
                "(InprocClient), not as a subprocess, or it re-executes "
                "ComfyUI's own main.py under multiprocessing.spawn and crashes. "
                "See this module's docstring.",
                _MULTIPROCESSING_ENV_VAR, original_value,
            )
        os.environ[_MULTIPROCESSING_ENV_VAR] = "0"
        try:
            self._engine = LLMEngine.from_engine_args(engine_args_obj)
        finally:
            if original_value is None:
                del os.environ[_MULTIPROCESSING_ENV_VAR]
            else:
                os.environ[_MULTIPROCESSING_ENV_VAR] = original_value
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
        # Best-effort shutdown before dropping the reference, so VRAM is
        # reclaimed on model swap (registry.py calls this on pipeline
        # replacement). The real shutdown path is self._engine.engine_core
        # (an InprocClient), not self._engine itself -- LLMEngine has no
        # shutdown() of its own. engine_core.shutdown() reaches vLLM's
        # gc.unfreeze(), which is required for weights/KV-cache to become
        # collectible at all. Falls back to older/alternate shutdown method
        # names for compatibility with other vLLM versions or client types.
        if self._engine is not None:
            shutdown = getattr(getattr(self._engine, "engine_core", None), "shutdown", None)
            if shutdown is None:
                shutdown = getattr(self._engine, "shutdown", None)
            if shutdown is None:
                shutdown = getattr(self._engine, "shutdown_background_loop", None)
            if shutdown is not None:
                shutdown()
        self._engine = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self) -> None:
        # Best-effort backstop: registry.py only calls unload() explicitly
        # under specific conditions (pipeline re-registered under the same
        # name, no active session) -- when those don't hold, this instance
        # is simply dereferenced with no explicit cleanup. __del__ catches
        # that case once CPython actually collects this instance. Not a
        # complete fix (no help if a reference is held indefinitely, or
        # under a reference cycle) -- a known cross-provider registry
        # limitation, not vLLM-specific.
        try:
            self.unload()
        except Exception:
            logger.warning("vllm provider: exception during __del__ cleanup", exc_info=True)

    def _drive_to_completion(
        self,
        request_id: str,
        prompt_text: str,
        sampling_params: SamplingParams,
        stop_event: threading.Event,
    ) -> Iterator[str]:
        """Runs on bridge_sync_iterator's worker thread: adds the request to
        the engine and drives it via a blocking step() loop, yielding
        incremental text deltas."""
        logger.debug(
            "vllm request %s: add_request, prompt length %d chars",
            request_id, len(prompt_text),
        )
        generated_text = ""
        step_count = 0
        finished_normally = False
        try:
            self._engine.add_request(request_id, prompt_text, sampling_params)
            while self._engine.has_unfinished_requests():
                if stop_event.is_set():
                    # Only reachable if cancelled before the first token --
                    # once an item has been yielded, cleanup happens in the
                    # finally: block below instead.
                    logger.warning(
                        "vllm request %s: cancelled before first token, aborting",
                        request_id,
                    )
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
                        finished_normally = True
                        return
        except Exception:
            # Log with the traceback here -- bridge_sync_iterator forwards
            # the exception to the caller, but doesn't log it.
            logger.error(
                "vllm request %s: unhandled exception after %d steps",
                request_id, step_count, exc_info=True,
            )
            raise
        finally:
            # abort_request() must live here, not in the stop_event branch
            # above: once this generator has yielded an item,
            # bridge_sync_iterator's own per-item stop_event check usually
            # abandons it (via GeneratorExit) before it can be resumed to
            # check stop_event itself. finally: runs on that path too, so
            # it's the one reliable place to call abort_request().
            if not finished_normally:
                logger.warning(
                    "vllm request %s: aborting after %d steps (cancelled, "
                    "abandoned, or errored)",
                    request_id, step_count,
                )
                self._engine.abort_request(request_id)

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

        # asyncio.to_thread avoids a raw blocking self._lock.acquire() here,
        # which would freeze the event loop while contended and could
        # deadlock against another generate() call holding the lock.
        await asyncio.to_thread(self._lock.acquire)
        bridge = bridge_sync_iterator(factory, stop_event)
        try:
            async for text in bridge:
                yield GenerationDelta(text=text, finished=False)
            yield GenerationDelta(text="", finished=True)
        finally:
            # Await aclose() (not just let the async-for exit) so the worker
            # thread is actually joined before the lock releases --
            # otherwise a second generate() could race the first call's
            # worker thread.
            await bridge.aclose()
            self._lock.release()

    def _build_chat_messages(self, messages: list[ChatMessage]) -> list[dict]:
        result = []
        if self._system_prompt:
            result.append({"role": "system", "content": self._system_prompt})
        result.extend({"role": m.role, "content": m.content} for m in messages)
        return result
