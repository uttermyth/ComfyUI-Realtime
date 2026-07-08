"""VLLMProvider tests run entirely against a stub synchronous LLMEngine (see
tests/_stub_vllm_engine.py) -- vLLM's real engine requires a CUDA GPU this
test suite cannot assume is present. Real engine/model behavior is manual
verification only, per
docs/superpowers/specs/2026-07-08-vllm-provider-redesign-design.md's Testing
Strategy section."""
import asyncio

import pytest

pytest.importorskip("vllm")

from comfyui_realtime.providers import vllm_llm
from comfyui_realtime.providers.base import ChatMessage, GenerationOptions
from comfyui_realtime.providers.vllm_llm import VLLMProvider
from tests._stub_vllm_engine import (
    NoChatTemplateTokenizer,
    StubTokenizer,
    make_stub_engine_class,
)


def _patch_engine_and_tokenizer(monkeypatch, *, engine_class=None, tokenizer=None):
    monkeypatch.setattr(vllm_llm, "LLMEngine", engine_class or make_stub_engine_class())
    monkeypatch.setattr(vllm_llm, "get_tokenizer", lambda *a, **k: tokenizer or StubTokenizer())


def test_construction_raises_if_no_cuda_gpu(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        VLLMProvider(model_path="/fake/path")


def test_construction_raises_if_tokenizer_has_no_chat_template(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    _patch_engine_and_tokenizer(monkeypatch, tokenizer=NoChatTemplateTokenizer())
    with pytest.raises(ValueError, match="chat_template"):
        VLLMProvider(model_path="/fake/path")


def test_construction_builds_engine_args_from_model_path(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class()
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)

    VLLMProvider(model_path="/fake/path", gpu_memory_utilization=0.5)

    (instance,) = StubEngine.instances
    assert instance.engine_args.model == "/fake/path"
    assert instance.engine_args.gpu_memory_utilization == 0.5


def test_construction_leaves_max_model_len_dtype_quantization_unset_by_default(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class()
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)

    VLLMProvider(model_path="/fake/path")

    (instance,) = StubEngine.instances
    default_args = type(instance.engine_args)(model="/fake/path")
    assert instance.engine_args.max_model_len == default_args.max_model_len
    assert instance.engine_args.dtype == default_args.dtype
    assert instance.engine_args.quantization == default_args.quantization


def test_construction_applies_explicit_max_model_len_dtype_quantization(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class()
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)

    VLLMProvider(
        model_path="/fake/path",
        max_model_len=4096,
        dtype="bfloat16",
        quantization="modelopt",
    )

    (instance,) = StubEngine.instances
    assert instance.engine_args.max_model_len == 4096
    assert instance.engine_args.dtype == "bfloat16"
    assert instance.engine_args.quantization == "modelopt"


def test_engine_args_json_overlay_applies_valid_overrides(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class()
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)

    VLLMProvider(model_path="/fake/path", engine_args='{"tensor_parallel_size": 2}')

    (instance,) = StubEngine.instances
    assert instance.engine_args.tensor_parallel_size == 2


def test_engine_args_invalid_json_raises_value_error(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    _patch_engine_and_tokenizer(monkeypatch)
    with pytest.raises(ValueError, match="not valid JSON"):
        VLLMProvider(model_path="/fake/path", engine_args="{not json")


def test_engine_args_unknown_key_raises_value_error_with_hint(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    _patch_engine_and_tokenizer(monkeypatch)
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        VLLMProvider(model_path="/fake/path", engine_args='{"tensor_paralel_size": 2}')


def test_unload_clears_engine_and_tokenizer(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class()
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)
    provider = VLLMProvider(model_path="/fake/path")
    (instance,) = StubEngine.instances

    provider.unload()

    assert instance.shutdown_call_count == 1
    assert provider._engine is None
    assert provider._tokenizer is None


def test_unload_is_safe_when_engine_has_no_shutdown_method(monkeypatch):
    """Some vLLM versions may not expose a shutdown()/shutdown_background_loop()
    method at all -- unload() must not crash in that case."""
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class()
    del StubEngine.shutdown
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)
    provider = VLLMProvider(model_path="/fake/path")

    provider.unload()

    assert provider._engine is None
    assert provider._tokenizer is None


async def test_generate_yields_incremental_deltas_then_a_finished_delta(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class(chunks=["Hello", "Hello world"])
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)
    provider = VLLMProvider(model_path="/fake/path")

    deltas = []
    async for delta in provider.generate(
        [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=16)
    ):
        deltas.append(delta)

    assert deltas[-1].finished is True
    assert deltas[-1].text == ""
    non_final = deltas[:-1]
    assert [d.text for d in non_final] == ["Hello", " world"]
    assert all(d.finished is False for d in non_final)


async def test_generate_builds_prompt_via_apply_chat_template_with_system_prompt(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class(chunks=["hi"])
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)
    provider = VLLMProvider(model_path="/fake/path", system_prompt="Be terse.")

    async for _ in provider.generate(
        [ChatMessage(role="user", content="hello")], GenerationOptions()
    ):
        pass

    (instance,) = StubEngine.instances
    (request_id, prompt, sampling_params) = instance.add_request_calls[0]
    assert prompt == "system: Be terse.\nuser: hello\nassistant:"
    assert isinstance(request_id, str) and request_id


async def test_generate_default_max_tokens_is_512_when_unset(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class(chunks=["hi"])
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)
    provider = VLLMProvider(model_path="/fake/path")

    async for _ in provider.generate(
        [ChatMessage(role="user", content="hello")], GenerationOptions(max_tokens=None)
    ):
        pass

    (instance,) = StubEngine.instances
    (_, _, sampling_params) = instance.add_request_calls[0]
    assert sampling_params.max_tokens == 512


async def test_generate_aborts_request_when_caller_stops_consuming_early(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class(chunks=["a", "ab", "abc", "abcd"])
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)
    provider = VLLMProvider(model_path="/fake/path")

    agen = provider.generate([ChatMessage(role="user", content="hi")], GenerationOptions())
    first_delta = await agen.__anext__()
    assert first_delta.text == "a"
    await agen.aclose()

    (instance,) = StubEngine.instances
    (request_id, _, _) = instance.add_request_calls[0]
    # Stopped early: not all 4 scripted chunks were consumed...
    assert instance.consumed_chunk_counts[request_id] < 4
    # ...and VLLMProvider explicitly called abort_request for this
    # request_id -- proving generate()'s stop_event-checked cancellation
    # path ran (this provider has no aclose()-triggers-internal-abort
    # behavior like the old AsyncLLMEngine version did -- it owns its own
    # step loop and must check stop_event itself).
    assert instance.abort_request_calls == [request_id]


async def test_aclose_waits_for_worker_thread_before_returning_so_lock_release_is_safe(monkeypatch):
    # Mirrors LlamaCppLLMProvider's equivalent guarantee: awaiting
    # agen.aclose() must not return until the underlying worker thread (and
    # its bridge_sync_iterator finally block) has actually finished, or a
    # second generate() call right after could race the first call's
    # worker thread against the same engine. delay=0.05 gives the worker
    # thread's step() calls enough real wall-clock time that a broken
    # implementation (releasing the lock too early) would have a realistic
    # window to let the second call's add_request() collide with the
    # first's still-running worker thread.
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class(chunks=["a", "ab", "abc"], delay=0.05)
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)
    provider = VLLMProvider(model_path="/fake/path")

    first = provider.generate([ChatMessage(role="user", content="one")], GenerationOptions())
    await first.__anext__()
    await first.aclose()

    second_deltas = []
    async for delta in provider.generate(
        [ChatMessage(role="user", content="two")], GenerationOptions()
    ):
        second_deltas.append(delta)
    assert second_deltas[-1].finished is True


async def test_two_concurrent_generate_calls_serialize(monkeypatch):
    # Inverse of this provider's old test proving AsyncLLMEngine-based
    # interleaving: this provider now holds a threading.Lock, so two
    # concurrent generate() calls on one instance must serialize instead --
    # every event from whichever call acquires the lock first must appear
    # before any event from the other, not interleaved. delay=0.01 gives
    # each stub step() a small real wall-clock cost so genuine interleaving
    # (if the lock somehow weren't blocking) would be observable rather
    # than accidentally passing due to both calls happening to run
    # back-to-back on the same thread.
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class(chunks=["a", "ab", "abc"], delay=0.01)
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)
    provider = VLLMProvider(model_path="/fake/path")

    events: list[str] = []

    async def consume(label: str) -> None:
        async for delta in provider.generate(
            [ChatMessage(role="user", content=label)], GenerationOptions()
        ):
            if not delta.finished:
                events.append(label)

    await asyncio.gather(consume("A"), consume("B"))

    assert events.count("A") == 3
    assert events.count("B") == 3
    half = len(events) // 2
    all_a_first = events[:half] == ["A"] * half
    all_b_first = events[:half] == ["B"] * half
    assert all_a_first or all_b_first, (
        f"expected one call's events to fully precede the other's (serialized), got {events}"
    )
