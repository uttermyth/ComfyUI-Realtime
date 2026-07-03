"""VLLMProvider loads a local HuggingFace transformers-format model
directory (same on-disk shape as TransformersLLMProvider -- config.json +
.safetensors weights + tokenizer files) via vLLM's AsyncLLMEngine.

Unlike LlamaCppLLMProvider and TransformersLLMProvider, this provider holds
no threading.Lock. AsyncLLMEngine is natively asyncio-based and performs
its own continuous batching across concurrently-submitted requests -- and
this pipeline genuinely produces concurrent load against one provider
instance: every realtime session naming the same pipeline shares one
ILLMProvider instance (see PipelineConfig.llm in registry.py), and nothing
above the provider layer assumes single-flight per provider instance, only
single-flight per session (enforced in server/websocket_handler.py,
independent of the provider). See
docs/superpowers/specs/2026-07-02-vllm-llm-provider-design.md's
"Concurrency Model & Pipeline Verification" section for the full analysis
this conclusion is based on.

Quantization (NVFP4/modelopt, FP8, AWQ, GPTQ, ...) is not a concept this
provider special-cases: vLLM auto-detects quant_method from the
checkpoint's own config.json at engine-construction time. The
`quantization` constructor parameter is only an explicit override for the
rare case a caller needs to force a specific method; leaving it empty lets
vLLM's own detection run.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
from typing import AsyncIterator

import torch
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
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
                "VLLMProvider requires a CUDA GPU (torch.cuda.is_available() is "
                "False). vLLM's AsyncLLMEngine does not support this provider's "
                "CPU/MPS fallback the way TransformersLLMProvider does -- use "
                "TransformersLLMProvider or LlamaCppLLMProvider on non-CUDA hardware."
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
        dataclasses.replace -- chosen (matching LocalAI's reference vLLM
        backend's identical _apply_engine_args helper) specifically because
        replace() re-runs AsyncEngineArgs.__post_init__, so dict-valued
        overrides get vLLM's own normal dataclass coercion rather than
        being passed through as raw dicts."""
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
        self._engine = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
