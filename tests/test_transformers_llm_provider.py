"""No real checkpoint download needed -- see tests/_tiny_transformers_fixture.py.
Not marked pytest.mark.integration (unlike test_llama_cpp_llm_provider.py /
test_pocket_tts_provider.py) since this provider's fixture model is built
from scratch locally and runs fast even on CPU."""
import pathlib
import threading

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

import torch

from comfyui_realtime.providers.base import ChatMessage, GenerationOptions
from comfyui_realtime.providers.transformers_llm import TransformersLLMProvider, _StopEventCriteria
from tests._tiny_transformers_fixture import build_tiny_transformers_model_dir


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> pathlib.Path:
    target_dir = tmp_path_factory.mktemp("tiny_transformers_model")
    build_tiny_transformers_model_dir(target_dir)
    return target_dir


@pytest.fixture(scope="module")
def tiny_model_dir_without_chat_template(tmp_path_factory) -> pathlib.Path:
    target_dir = tmp_path_factory.mktemp("tiny_transformers_model_no_template")
    build_tiny_transformers_model_dir(target_dir, include_chat_template=False)
    return target_dir


def test_construction_succeeds_on_cpu(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        assert hasattr(provider, "generate")
    finally:
        provider.unload()


def test_construction_raises_when_chat_template_missing(tiny_model_dir_without_chat_template):
    with pytest.raises(ValueError, match="chat_template"):
        TransformersLLMProvider(model_path=str(tiny_model_dir_without_chat_template), device="cpu")


def test_construction_raises_for_unavailable_cuda_device(tiny_model_dir, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        TransformersLLMProvider(model_path=str(tiny_model_dir), device="cuda")


def test_construction_raises_for_unavailable_mps_device(tiny_model_dir, monkeypatch):
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="mps"):
        TransformersLLMProvider(model_path=str(tiny_model_dir), device="mps")


def test_auto_dtype_resolves_to_float32_on_cpu(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu", torch_dtype="auto")
    try:
        assert next(provider._model.parameters()).dtype == torch.float32
    finally:
        provider.unload()


def test_explicit_dtype_is_respected(tiny_model_dir):
    provider = TransformersLLMProvider(
        model_path=str(tiny_model_dir), device="cpu", torch_dtype="bfloat16"
    )
    try:
        assert next(provider._model.parameters()).dtype == torch.bfloat16
    finally:
        provider.unload()


def test_trust_remote_code_defaults_to_false(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        assert provider._trust_remote_code is False
    finally:
        provider.unload()


def test_unload_releases_the_model(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    provider.unload()
    assert provider._model is None


def test_stop_event_criteria_returns_false_until_event_is_set():
    event = threading.Event()
    criteria = _StopEventCriteria(event)
    assert criteria(None, None) is False
    event.set()
    assert criteria(None, None) is True


async def test_generate_yields_text_deltas_then_a_finished_delta(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        messages = [ChatMessage(role="user", content="hello world")]
        deltas = []
        async for delta in provider.generate(messages, GenerationOptions(max_tokens=8)):
            deltas.append(delta)
        assert len(deltas) >= 1
        assert deltas[-1].finished is True
        assert deltas[-1].text == ""
        assert all(d.finished is False for d in deltas[:-1])
    finally:
        provider.unload()


async def test_system_prompt_is_included_in_the_formatted_prompt(tiny_model_dir, monkeypatch):
    provider = TransformersLLMProvider(
        model_path=str(tiny_model_dir), device="cpu", system_prompt="banana"
    )
    try:
        captured = {}
        original_apply = provider._tokenizer.apply_chat_template

        def spy_apply_chat_template(messages, **kwargs):
            captured["messages"] = messages
            return original_apply(messages, **kwargs)

        monkeypatch.setattr(provider._tokenizer, "apply_chat_template", spy_apply_chat_template)

        async for _ in provider.generate(
            [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=4)
        ):
            pass

        assert captured["messages"][0] == {"role": "system", "content": "banana"}
    finally:
        provider.unload()


async def test_temperature_zero_uses_greedy_decoding(tiny_model_dir, monkeypatch):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        captured_kwargs = {}
        original_generate = provider._model.generate

        def spy_generate(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_generate(*args, **kwargs)

        monkeypatch.setattr(provider._model, "generate", spy_generate)

        async for _ in provider.generate(
            [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=4, temperature=0.0)
        ):
            pass

        assert captured_kwargs["do_sample"] is False
        assert "temperature" not in captured_kwargs
    finally:
        provider.unload()


async def test_positive_temperature_uses_sampling(tiny_model_dir, monkeypatch):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        captured_kwargs = {}
        original_generate = provider._model.generate

        def spy_generate(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_generate(*args, **kwargs)

        monkeypatch.setattr(provider._model, "generate", spy_generate)

        async for _ in provider.generate(
            [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=4, temperature=0.7)
        ):
            pass

        assert captured_kwargs["do_sample"] is True
        assert captured_kwargs["temperature"] == 0.7
    finally:
        provider.unload()


async def test_default_max_new_tokens_is_512_when_unspecified(tiny_model_dir, monkeypatch):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        captured_kwargs = {}
        original_generate = provider._model.generate

        def spy_generate(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_generate(*args, **kwargs)

        monkeypatch.setattr(provider._model, "generate", spy_generate)

        async for _ in provider.generate(
            [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=None)
        ):
            pass

        assert captured_kwargs["max_new_tokens"] == 512
    finally:
        provider.unload()


async def test_abandoning_generation_stops_the_inner_generate_thread(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        gen = provider.generate(
            [ChatMessage(role="user", content="hello world")], GenerationOptions(max_tokens=64)
        )
        await gen.__anext__()
        await gen.aclose()

        inner_threads = [
            t for t in threading.enumerate() if t.name == "transformers-generate-inner" and t.is_alive()
        ]
        assert inner_threads == [], "inner model.generate() thread is still alive after aclose()"
    finally:
        provider.unload()


async def test_abandoning_generation_releases_the_lock_promptly(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        first_gen = provider.generate(
            [ChatMessage(role="user", content="hello world")], GenerationOptions(max_tokens=64)
        )
        await first_gen.__anext__()
        await first_gen.aclose()

        second_deltas = []
        async for delta in provider.generate(
            [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=4)
        ):
            second_deltas.append(delta)
        assert len(second_deltas) >= 1
    finally:
        provider.unload()


@pytest.mark.timeout(10)
async def test_exception_inside_generate_propagates_instead_of_hanging(tiny_model_dir, monkeypatch):
    # Regression test: threading.Thread silently swallows exceptions raised
    # in its target. Without generate()'s own capture-and-reraise, this
    # would hang forever (the streamer never gets its terminating sentinel)
    # instead of raising -- see Task 3's "Update" note above. @pytest.mark.timeout(10)
    # (from pytest-timeout, added in Step 1 above) makes a regression back to
    # that hang fail this one test after 10s instead of hanging the whole
    # test run indefinitely.
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        def broken_generate(*args, **kwargs):
            raise RuntimeError("simulated generation failure")

        monkeypatch.setattr(provider._model, "generate", broken_generate)

        with pytest.raises(RuntimeError, match="simulated generation failure"):
            async for _ in provider.generate(
                [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=4)
            ):
                pass
    finally:
        provider.unload()
