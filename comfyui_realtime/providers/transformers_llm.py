"""TransformersLLMProvider loads a local HuggingFace transformers-format
chat model (config.json + .safetensors weights + tokenizer files) --
v1 plain fp16/bf16/fp32 only, no on-the-fly quantization, no device_map
sharding (see docs/superpowers/specs/2026-07-01-transformers-llm-provider-design.md).

Device/dtype are resolved once at construction and fixed for the life of
this instance -- "auto" picks cuda > mps > cpu, and an explicitly-requested
device that isn't actually available raises RuntimeError immediately
rather than failing confusingly mid-`.to(device)`.

Tokenizer must have a chat_template (checked at construction) -- v1 only
supports instruct/chat-tuned models, not base/completion-only models.

generate()'s streaming/cancellation design is documented in
transformers_llm.py's generate() docstring, added in a later commit --
see the design spec for the full reasoning (model.generate() is
push/blocking, unlike llama.cpp's pull-based streaming, so cancellation
needs an explicit StoppingCriteria rather than "stop calling next()").
"""
from __future__ import annotations

import threading
from typing import AsyncIterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ChatMessage, GenerationDelta, GenerationOptions

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested but torch.cuda.is_available() is False.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' was requested but torch.backends.mps.is_available() is False.")
    return device


def _resolve_dtype(torch_dtype: str, device: str) -> torch.dtype:
    if torch_dtype == "auto":
        return torch.float32 if device == "cpu" else torch.float16
    return _DTYPE_MAP[torch_dtype]


class TransformersLLMProvider:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        system_prompt: str = "",
    ) -> None:
        resolved_device = _resolve_device(device)
        resolved_dtype = _resolve_dtype(torch_dtype, resolved_device)
        self._trust_remote_code = trust_remote_code

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.chat_template is None:
            raise ValueError(
                f"Model at {model_path!r} has no tokenizer chat_template. "
                "TransformersLLMProvider only supports instruct/chat-tuned models "
                "with a chat template -- base/completion-only models aren't "
                "supported in v1."
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=resolved_dtype, trust_remote_code=trust_remote_code
        )
        self._model.to(resolved_device)
        self._model.eval()

        self._device = resolved_device
        self._system_prompt = system_prompt
        self._lock = threading.Lock()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._device == "cuda":
            torch.cuda.empty_cache()
        elif self._device == "mps":
            torch.mps.empty_cache()

    async def generate(
        self, messages: list[ChatMessage], options: GenerationOptions
    ) -> AsyncIterator[GenerationDelta]:
        """Streaming generation -- implemented in Task 3 (see this module's
        docstring for why model.generate()'s push/blocking streaming needs a
        StoppingCriteria-based cancellation design rather than llama.cpp's
        pull-based generator)."""
        raise NotImplementedError("TransformersLLMProvider.generate() is implemented in Task 3")
        yield  # pragma: no cover -- unreachable; makes this an async generator function
