"""Test-only stand-ins for vLLM's AsyncLLMEngine and tokenizer. vLLM's
AsyncLLMEngine effectively requires a CUDA GPU to do anything real, which
no CI/dev box running this test suite is guaranteed to have -- these tests
exercise VLLMProvider's own logic (chat formatting, cumulative-to-incremental
delta conversion, engine_args overlay, cancellation, concurrency) against a
stub engine rather than a real one. Real engine construction, real model
loading, and real NVFP4/quantized-checkpoint behavior are manual
verification only -- see the design spec's Testing Strategy section."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class StubCompletionOutput:
    text: str


@dataclass
class StubRequestOutput:
    outputs: list[StubCompletionOutput]


class StubTokenizer:
    chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        rendered = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        return rendered + "assistant:"


class NoChatTemplateTokenizer(StubTokenizer):
    chat_template = None


def make_stub_engine_class(chunks: list[str] | None = None, delay: float = 0.0) -> type:
    """Returns a fresh class (not instance) each call, shaped like vLLM's
    AsyncLLMEngine: .from_engine_args(engine_args) constructs an instance
    that records what it was built with, and .generate(prompt,
    sampling_params, request_id) yields a scripted sequence of cumulative
    text chunks -- mirroring vLLM's real RequestOutput.outputs[0].text
    semantics (each yield carries the FULL text so far, not a delta).
    delay > 0 inserts an await asyncio.sleep(delay) before each yield, used
    by the concurrency test (Task 5) to force two concurrent generate()
    calls to interleave rather than happening to run back-to-back by
    accident."""
    scripted_chunks = chunks if chunks is not None else ["Hello", "Hello world"]

    class _StubAsyncLLMEngine:
        instances: list["_StubAsyncLLMEngine"] = []

        def __init__(self, engine_args) -> None:
            self.engine_args = engine_args
            self.generate_calls: list[tuple[str, object, str]] = []
            self.consumed_chunk_counts: dict[str, int] = {}
            self.finished_request_ids: list[str] = []
            _StubAsyncLLMEngine.instances.append(self)

        @classmethod
        def from_engine_args(cls, engine_args):
            return cls(engine_args)

        async def generate(self, prompt, sampling_params, request_id):
            self.generate_calls.append((prompt, sampling_params, request_id))
            self.consumed_chunk_counts[request_id] = 0
            try:
                for chunk in scripted_chunks:
                    if delay:
                        await asyncio.sleep(delay)
                    self.consumed_chunk_counts[request_id] += 1
                    yield StubRequestOutput(outputs=[StubCompletionOutput(text=chunk)])
            finally:
                # Mirrors vLLM's own AsyncLLMEngine.generate(), which aborts
                # the request in its own finally block regardless of
                # whether the caller consumed the generator to exhaustion
                # or closed it early -- this stub only needs to prove
                # VLLMProvider's finally: await result_generator.aclose()
                # actually reaches this point, which is exactly what the
                # cancellation test (Task 4) asserts on.
                self.finished_request_ids.append(request_id)

    return _StubAsyncLLMEngine
