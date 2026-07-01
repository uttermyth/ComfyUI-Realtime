"""Requires the lessac voice files under assets/piper_voices/en_US/lessac/."""
import pathlib

import pytest

pytest.importorskip("piper")

from comfyui_realtime.providers.piper_tts import PiperTTSProvider

VOICES_DIR = pathlib.Path(__file__).parent.parent / "assets" / "piper_voices" / "en_US" / "lessac"

pytestmark = pytest.mark.integration


def _voice_paths(quality: str) -> tuple[str, str]:
    onnx = VOICES_DIR / quality / f"en_US-lessac-{quality}.onnx"
    config = VOICES_DIR / quality / f"en_US-lessac-{quality}.onnx.json"
    assert onnx.exists(), f"missing {onnx}"
    assert config.exists(), f"missing {config}"
    return str(onnx), str(config)


@pytest.fixture(scope="module")
def provider():
    provider = PiperTTSProvider(
        voices={
            "lessac-medium": _voice_paths("medium"),  # native 22.05kHz
            "lessac-low": _voice_paths("low"),  # native 16kHz
        },
        default_voice="lessac-medium",
    )
    yield provider
    provider.unload()


async def _text_stream(*chunks: str):
    for chunk in chunks:
        yield chunk


async def test_synthesize_default_voice_yields_nonempty_audio(provider):
    audio_chunks = [chunk async for chunk in provider.synthesize(_text_stream("Hello there."))]
    assert len(audio_chunks) > 0
    assert sum(len(c) for c in audio_chunks) > 0
    assert all(len(c) % 2 == 0 for c in audio_chunks)  # valid pcm16


async def test_synthesize_low_quality_voice_also_works(provider):
    audio_chunks = [
        chunk async for chunk in provider.synthesize(_text_stream("Hello there."), voice="lessac-low")
    ]
    assert sum(len(c) for c in audio_chunks) > 0


async def test_synthesize_handles_multiple_text_chunks_in_order(provider):
    audio_chunks = [
        chunk async for chunk in provider.synthesize(_text_stream("First sentence.", "Second sentence."))
    ]
    assert sum(len(c) for c in audio_chunks) > 0


def test_list_voices_reports_both_registered_voices(provider):
    voice_ids = {v.id for v in provider.list_voices()}
    assert voice_ids == {"lessac-medium", "lessac-low"}


def test_output_sample_rate_is_the_wire_rate(provider):
    assert provider.output_sample_rate == 24000
    assert provider.output_format == "pcm16"


async def test_abandoning_synthesis_releases_the_model_lock_promptly(provider):
    # Regression test for the nested-async-generator lock-release hazard
    # Task 3 found in LlamaCppLLMProvider: explicitly aclose() a synthesize()
    # call mid-stream, then immediately start a second one. If the lock from
    # the first call weren't released until its worker thread actually
    # joined, this would race two Piper calls on the same voice's ONNX
    # session (or simply hang/take far longer than one synthesis call).
    import time

    first_gen = provider.synthesize(_text_stream("First sentence.", "Second sentence."))
    await first_gen.__anext__()
    await first_gen.aclose()

    start = time.perf_counter()
    second_chunks = [chunk async for chunk in provider.synthesize(_text_stream("Quick."))]
    elapsed = time.perf_counter() - start

    assert sum(len(c) for c in second_chunks) > 0
    assert elapsed < 5.0, f"second synthesize() took {elapsed:.2f}s -- lock from first call may not have released"
