"""Requires models/qwen2.5-0.5b-instruct-q8_0.gguf (repo root)."""
import pathlib
import threading
import time

import pytest

pytest.importorskip("llama_cpp")

from comfyui_realtime.providers.base import ChatMessage, GenerationOptions
from comfyui_realtime.providers.llama_cpp_llm import LlamaCppLLMProvider

MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def provider():
    assert MODEL_PATH.exists(), f"test model not found at {MODEL_PATH}"
    provider = LlamaCppLLMProvider(model_path=str(MODEL_PATH), n_ctx=512)
    yield provider
    provider.unload()


async def test_generate_yields_text_deltas_then_a_finished_delta(provider):
    messages = [ChatMessage(role="user", content="Say hello in exactly one word.")]
    deltas = []
    async for delta in provider.generate(messages, GenerationOptions(max_tokens=16)):
        deltas.append(delta)
    assert len(deltas) > 1
    assert deltas[-1].finished is True
    assert deltas[-1].text == ""
    assert all(d.finished is False for d in deltas[:-1])
    full_text = "".join(d.text for d in deltas[:-1])
    assert len(full_text.strip()) > 0


async def test_system_prompt_is_included():
    provider = LlamaCppLLMProvider(
        model_path=str(MODEL_PATH), n_ctx=512, system_prompt="Reply only with the word BANANA."
    )
    try:
        deltas = []
        async for delta in provider.generate(
            [ChatMessage(role="user", content="What is your favorite fruit?")],
            GenerationOptions(max_tokens=16, temperature=0.0),
        ):
            deltas.append(delta)
        full_text = "".join(d.text for d in deltas if not d.finished)
        assert "banana" in full_text.lower()
    finally:
        provider.unload()


async def test_abandoning_generation_releases_the_model_lock_promptly(provider):
    # Start a generation, abandon it after one delta, then immediately start
    # a second one -- if the lock from the first call weren't released
    # promptly, this would hang or take much longer than one model's worth
    # of latency (spec section 6.6/7.3: one inference at a time per model,
    # released at the next boundary on abandonment).
    messages = [ChatMessage(role="user", content="Write a long story about a journey.")]

    first_gen = provider.generate(messages, GenerationOptions(max_tokens=256))
    await first_gen.__anext__()
    await first_gen.aclose()

    start = time.perf_counter()
    second_deltas = []
    async for delta in provider.generate(
        [ChatMessage(role="user", content="Say hi.")], GenerationOptions(max_tokens=8)
    ):
        second_deltas.append(delta)
    elapsed = time.perf_counter() - start

    assert len(second_deltas) > 1
    assert elapsed < 5.0, f"second generate() took {elapsed:.2f}s -- lock from first call may not have released"
