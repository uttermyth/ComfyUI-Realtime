"""Tests SileroVADProvider against synthesized speech (via Phase 1's
PiperTTSProvider, resampled to 16kHz) and synthetic silence -- no external
audio fixture needed."""
import pathlib

import pytest

pytest.importorskip("silero_vad")
pytest.importorskip("piper")

from comfyui_realtime.engine.resample import resample_pcm16
from comfyui_realtime.providers.piper_tts import PiperTTSProvider
from comfyui_realtime.providers.silero_vad import SileroVADProvider

VOICES_DIR = pathlib.Path(__file__).parent.parent / "assets" / "piper_voices" / "en_US" / "lessac"

pytestmark = pytest.mark.integration

FRAME_BYTES = 512 * 2  # 512 samples, 16-bit pcm


def _chunk_exact(audio: bytes, frame_bytes: int) -> list[bytes]:
    return [audio[i : i + frame_bytes] for i in range(0, len(audio) - frame_bytes + 1, frame_bytes)]


async def _synthesize_16k_speech(text: str) -> bytes:
    provider = PiperTTSProvider(
        voices={
            "lessac-medium": (
                str(VOICES_DIR / "medium" / "en_US-lessac-medium.onnx"),
                str(VOICES_DIR / "medium" / "en_US-lessac-medium.onnx.json"),
            )
        },
        default_voice="lessac-medium",
    )
    try:
        async def text_stream():
            yield text

        chunks = [chunk async for chunk in provider.synthesize(text_stream())]
        audio_24k = b"".join(chunks)
        return resample_pcm16(audio_24k, from_rate=24000, to_rate=16000)
    finally:
        provider.unload()


@pytest.fixture(scope="module")
def vad():
    provider = SileroVADProvider()
    yield provider
    provider.unload()


async def test_silence_never_triggers_speech_started(vad):
    silence = b"\x00\x00" * 512
    for _ in range(20):
        result = await vad.analyze(silence)
        assert result.speech_started is False


async def test_synthesized_speech_triggers_speech_started_then_ended(vad):
    speech_16k = await _synthesize_16k_speech("This is a test of voice activity detection.")
    frames = _chunk_exact(speech_16k, FRAME_BYTES)
    assert len(frames) > 10, "synthesized speech too short to test VAD boundaries meaningfully"

    saw_started = False
    saw_ended = False
    for frame in frames:
        result = await vad.analyze(frame)
        if result.speech_started:
            saw_started = True
        if result.speech_ended:
            saw_ended = True
    # Flush trailing silence so a boundary right at the end is still caught.
    for _ in range(10):
        result = await vad.analyze(b"\x00\x00" * 512)
        if result.speech_ended:
            saw_ended = True

    assert saw_started, "VAD never detected speech start in synthesized speech"
    assert saw_ended, "VAD never detected speech end after synthesized speech"


async def test_analyze_returns_a_probability_on_every_call(vad):
    silence = b"\x00\x00" * 512
    result = await vad.analyze(silence)
    assert 0.0 <= result.speech_probability <= 1.0


async def test_analyze_works_when_model_loaded_under_inference_mode():
    """Regression test (Task 11): ComfyUI's execution.py runs every node's
    load() call (including this provider's own loader) inside
    torch.inference_mode(). That marks the model's weights as inference
    tensors. A live end-to-end run against a real ComfyUI server (where
    the provider is constructed by SileroVADProviderNode during /prompt
    execution) crashed on the second analyze() call with "Inference
    tensors cannot be saved for backward", because the direct
    self._model(...) probability call wasn't itself wrapped in
    inference_mode -- only VADIterator's internal call was. Standalone
    tests that build SileroVADProvider() outside any inference_mode
    context never exercised this path."""
    import torch

    with torch.inference_mode():
        provider = SileroVADProvider()
    try:
        silence = b"\x00\x00" * 512
        for _ in range(5):
            result = await provider.analyze(silence)
            assert 0.0 <= result.speech_probability <= 1.0
    finally:
        provider.unload()
