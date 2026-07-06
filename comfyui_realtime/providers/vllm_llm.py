"""VLLMProvider loads a local HuggingFace transformers-format model
directory (same on-disk shape as TransformersLLMProvider -- config.json +
.safetensors weights + tokenizer files) via vLLM's AsyncLLMEngine.

Unlike LlamaCppLLMProvider and TransformersLLMProvider, this provider holds
no threading.Lock. AsyncLLMEngine is natively asyncio-based and performs
its own continuous batching across concurrently-submitted requests -- and
this pipeline genuinely produces concurrent load against one provider
instance: every realtime session naming the same pipeline shares one
ILLMProvider instance and nothing above the provider layer assumes 
single-flight per provider instance, only single-flight per session.

vLLM auto-detects quant_method from the
checkpoint's own config.json at engine-construction time. The
`quantization` constructor parameter is only an explicit override for the
rare case a caller needs to force a specific method; leaving it empty lets
vLLM's own detection run.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
import uuid
from typing import AsyncIterator

import torch
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

try:
    # vllm >= 0.22 moved tokenizer loading out of transformers_utils and into
    # its own top-level package.
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer

from .base import ChatMessage, GenerationDelta, GenerationOptions


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

        engine_args_obj = AsyncEngineArgs(
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

        self._engine = AsyncLLMEngine.from_engine_args(engine_args_obj)
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
        # Best-effort AsyncLLMEngine shutdown before dropping the reference.
        # AsyncLLMEngine runs a background engine loop (and in vLLM V1,
        # potentially an out-of-process EngineCore holding the model's
        # VRAM) -- simply letting Python garbage-collect self._engine does
        # not reliably stop that loop/process, and torch.cuda.empty_cache()
        # alone only returns already-freed allocator blocks, not VRAM a
        # still-alive engine holds. This matters because the registry calls
        # unload() specifically to reclaim resources when a pipeline is
        # replaced/unregistered (see registry.py's
        # _unload_orphaned_providers_locked) -- skipping a real shutdown
        # would leak a full model's worth of VRAM plus background
        # threads/processes on every pipeline swap. vLLM's shutdown/close
        # API has moved across versions, so this tries known method names
        # defensively rather than assuming one
        # exact name; it's a deliberate no-op if the installed version
        # exposes neither.
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
        result_generator = self._engine.generate(prompt_text, sampling_params, request_id)
        try:
            generated_text = ""
            async for output in result_generator:
                candidate_text = output.outputs[0].text
                delta = candidate_text.removeprefix(generated_text)
                generated_text = candidate_text
                if delta:
                    yield GenerationDelta(text=delta, finished=False)
            yield GenerationDelta(text="", finished=True)
        finally:
            # Closing the async generator vLLM itself returned -- rather
            # than calling self._engine.abort(request_id) directly --
            # delegates to vLLM's own internal abort-on-close handling.
            # Runs on both the normal-completion path and
            # an early-cancellation path
            await result_generator.aclose()

    def _build_chat_messages(self, messages: list[ChatMessage]) -> list[dict]:
        result = []
        if self._system_prompt:
            result.append({"role": "system", "content": self._system_prompt})
        result.extend({"role": m.role, "content": m.content} for m in messages)
        return result
