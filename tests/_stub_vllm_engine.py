"""Test-only stand-ins for vLLM's synchronous LLMEngine and tokenizer.
vLLM's LLMEngine effectively requires a CUDA GPU to do anything real, which
no CI/dev box running this test suite is guaranteed to have -- these tests
exercise VLLMProvider's own logic (chat formatting, cumulative-to-incremental
delta conversion, engine_args overlay, cancellation, serialization) against a
stub engine rather than a real one. Real engine construction, real model
loading, and real NVFP4/quantized-checkpoint behavior are manual
verification only -- see
docs/superpowers/specs/2026-07-08-vllm-provider-redesign-design.md's Testing
Strategy section.

Unlike the old AsyncLLMEngine-shaped stub this replaces, this one mimics
vLLM's SYNCHRONOUS LLMEngine surface: add_request()/step()/
has_unfinished_requests()/abort_request() are all plain blocking methods,
not coroutines -- step() must be called repeatedly by the caller (that's
VLLMProvider._drive_to_completion's job, not this stub's), and each step()
call returns a list of whatever RequestOutputs were produced that step
(exactly one per pending request per call here, since only one request is
ever added at a time under VLLMProvider's single-flight lock)."""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class StubCompletionOutput:
    text: str


@dataclass
class StubRequestOutput:
    request_id: str
    outputs: list[StubCompletionOutput]
    finished: bool = False


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
    synchronous LLMEngine: .from_engine_args(engine_args) constructs an
    instance that records what it was built with. .add_request(request_id,
    prompt, sampling_params) registers a pending request; .step() advances
    every pending request by exactly one scripted chunk and returns the
    list of StubRequestOutputs produced this step (mirroring vLLM's real
    RequestOutput.outputs[0].text semantics: each one carries the FULL text
    so far, not a delta); .has_unfinished_requests() reports whether any
    request is still pending; .abort_request(request_id) drops a pending
    request early.

    delay > 0 inserts a blocking time.sleep(delay) inside step() before it
    returns -- used by the lock-release and serialization tests (this file's
    Task 1 tests) to give a concurrently-started second generate() call a
    real window to (wrongly) interleave in, if the lock were not actually
    blocking it. This must be time.sleep, not asyncio.sleep: step() now runs
    synchronously on bridge_sync_iterator's worker thread, not as a
    coroutine on the event loop."""
    scripted_chunks = chunks if chunks is not None else ["Hello", "Hello world"]

    class _StubLLMEngine:
        instances: list["_StubLLMEngine"] = []

        def __init__(self, engine_args) -> None:
            self.engine_args = engine_args
            self.add_request_calls: list[tuple[str, object, object]] = []
            self.abort_request_calls: list[str] = []
            self.consumed_chunk_counts: dict[str, int] = {}
            self.shutdown_call_count = 0
            self._next_index: dict[str, int] = {}
            _StubLLMEngine.instances.append(self)

        @classmethod
        def from_engine_args(cls, engine_args):
            return cls(engine_args)

        def shutdown(self) -> None:
            # Mirrors the best-effort shutdown() method some vLLM LLMEngine
            # versions may expose -- VLLMProvider.unload() calls this (if
            # present) before dropping its engine reference, so tests can
            # assert it was actually invoked rather than just trusting the
            # reference got nulled out.
            self.shutdown_call_count += 1

        def add_request(self, request_id, prompt, sampling_params) -> None:
            self.add_request_calls.append((request_id, prompt, sampling_params))
            self.consumed_chunk_counts[request_id] = 0
            self._next_index[request_id] = 0

        def has_unfinished_requests(self) -> bool:
            return len(self._next_index) > 0

        def abort_request(self, request_id) -> None:
            self.abort_request_calls.append(request_id)
            self._next_index.pop(request_id, None)

        def step(self) -> list[StubRequestOutput]:
            outputs: list[StubRequestOutput] = []
            for request_id in list(self._next_index):
                if delay:
                    time.sleep(delay)
                index = self._next_index[request_id]
                if index >= len(scripted_chunks):
                    continue
                text = scripted_chunks[index]
                is_finished = index == len(scripted_chunks) - 1
                self.consumed_chunk_counts[request_id] += 1
                outputs.append(
                    StubRequestOutput(
                        request_id=request_id,
                        outputs=[StubCompletionOutput(text=text)],
                        finished=is_finished,
                    )
                )
                if is_finished:
                    del self._next_index[request_id]
                else:
                    self._next_index[request_id] = index + 1
            return outputs

    return _StubLLMEngine
