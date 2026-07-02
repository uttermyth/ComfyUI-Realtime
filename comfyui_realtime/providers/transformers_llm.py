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

import json
import logging
import os
import threading
from typing import AsyncIterator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from ..engine.executor_bridge import bridge_sync_iterator
from .base import ChatMessage, GenerationDelta, GenerationOptions

logger = logging.getLogger("comfyui_realtime")

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class _StopEventCriteria(StoppingCriteria):
    """Checked by model.generate() every decode step. Wired to the same
    stop_event bridge_sync_iterator manages, so an abandoned generate()
    call actually halts early instead of running to completion in the
    background -- see the module docstring for why this is necessary
    (model.generate() is push/blocking, not pull-based like llama.cpp)."""

    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._event.is_set()


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


def _warn_if_quantization_was_dropped(model_path: str, model) -> None:
    """transformers gracefully dequantizes a pre-quantized checkpoint (e.g.
    FP8) to full precision when no qualifying GPU/XPU is available, rather
    than raising -- see quantizers/quantizer_finegrained_fp8.py's
    validate_environment() and base.py's remove_quantization_config(),
    confirmed via tests/test_transformers_llm_provider.py's
    test_fp8_quantization_config_in_checkpoint_is_auto_detected. That's
    silent from this provider's perspective: the in-memory model's own
    quantization_config has already been stripped by the time
    from_pretrained() returns, so there's nothing left on `model` itself to
    check against -- the checkpoint's own config.json has to be read
    directly to know it was originally declared quantized. Sub-Compute-
    Capability-9 GPU behavior and missing-triton/compressed-tensors behavior
    on a qualifying GPU are NOT covered by this check -- both remain
    unverified, see the design doc addendum."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path) as f:
        declared_config = json.load(f)
    was_declared_quantized = "quantization_config" in declared_config
    is_actually_quantized = getattr(model, "is_quantized", False)
    if was_declared_quantized and not is_actually_quantized:
        logger.warning(
            "Model at %r declares a quantization_config in its config.json but "
            "loaded as a full-precision (dequantized) model -- this is expected "
            "on hardware without a qualifying GPU/XPU (e.g. FP8 quantization "
            "requires Compute Capability >= 9 / Hopper or newer). Memory usage "
            "and generation speed will match the full-precision model size, not "
            "the quantized size.",
            model_path,
        )


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
            model_path, torch_dtype=resolved_dtype, trust_remote_code=trust_remote_code
        )
        _warn_if_quantization_was_dropped(model_path, self._model)
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
        chat_messages = self._build_chat_messages(messages)
        # return_dict=True is required, not optional: apply_chat_template's
        # return type when return_dict is omitted has drifted across
        # transformers versions (some return a bare input_ids tensor, others
        # a BatchEncoding) -- return_dict=True is the documented, stable way
        # to always get a dict back. Passing attention_mask explicitly (not
        # just input_ids) also avoids generate() having to guess it from
        # padding, which misbehaves when pad_token == eos_token (common on
        # chat models).
        encoded = self._tokenizer.apply_chat_template(
            chat_messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        if options.temperature <= 0:
            sampling_kwargs = {"do_sample": False}
        else:
            sampling_kwargs = {"do_sample": True, "temperature": options.temperature}
        max_new_tokens = options.max_tokens or 512

        stop_event = threading.Event()

        def factory():
            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            # model.generate() runs on a second, inner thread (TextIteratorStreamer
            # is a producer/consumer design: generate() must run off the thread
            # that's iterating the streamer). threading.Thread silently swallows
            # any exception raised in its target -- without capturing it
            # ourselves, ANY error inside generate() (OOM, a bad input, a future
            # transformers version change) would leave the streamer with no
            # terminating sentinel, and the `for text in streamer` loop below
            # would hang forever with no error surfaced anywhere. generation_error
            # captures it (write-once by this thread, read-only-after-join() by
            # the code below -- inner_thread.join() is a memory barrier, so no
            # lock is needed). streamer.end() in the finally unconditionally
            # unblocks the loop -- it's the same method generate() already calls
            # itself on its normal-completion path, so calling it a second time
            # here is harmless (an extra, unread sentinel left in the queue after
            # the consumer has already broken out on the first one).
            generation_error: list[Exception] = []

            def _run_generate():
                try:
                    self._model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        streamer=streamer,
                        max_new_tokens=max_new_tokens,
                        stopping_criteria=StoppingCriteriaList([_StopEventCriteria(stop_event)]),
                        **sampling_kwargs,
                    )
                except Exception as exc:  # noqa: BLE001 -- forwarded below, not swallowed
                    generation_error.append(exc)
                finally:
                    streamer.end()

            inner_thread = threading.Thread(
                target=_run_generate, name="transformers-generate-inner", daemon=True
            )
            inner_thread.start()
            try:
                for text in streamer:
                    yield text
            finally:
                inner_thread.join()
            # Re-raising here (inside factory(), which runs on
            # bridge_sync_iterator's own worker thread) means this flows into
            # that module's EXISTING `except Exception as exc: queue.put_nowait(exc)`
            # forwarding path -- no new error-forwarding mechanism needed.
            if generation_error:
                raise generation_error[0]

        self._lock.acquire()
        bridge = bridge_sync_iterator(factory, stop_event)
        try:
            async for text in bridge:
                yield GenerationDelta(text=text, finished=False)
            yield GenerationDelta(text="", finished=True)
        finally:
            await bridge.aclose()
            self._lock.release()

    def _build_chat_messages(self, messages: list[ChatMessage]) -> list[dict]:
        result = []
        if self._system_prompt:
            result.append({"role": "system", "content": self._system_prompt})
        result.extend({"role": m.role, "content": m.content} for m in messages)
        return result
