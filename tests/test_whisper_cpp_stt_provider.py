"""Tests WhisperCppSTTProvider against synthesized speech (via Phase 1's
PiperTTSProvider, resampled to 16kHz) -- no external audio fixture needed."""
import pathlib

import pytest

pytest.importorskip("pywhispercpp")
pytest.importorskip("piper")

from comfyui_realtime.engine.resample import resample_pcm16
from comfyui_realtime.providers.piper_tts import PiperTTSProvider
from comfyui_realtime.providers.whisper_cpp_stt import WhisperCppSTTProvider

VOICES_DIR = pathlib.Path(__file__).parent.parent / "assets" / "piper_voices" / "en_US" / "lessac"
MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "ggml-base.en.bin"

pytestmark = pytest.mark.integration


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
def stt():
    assert MODEL_PATH.exists(), f"test model not found at {MODEL_PATH}"
    provider = WhisperCppSTTProvider(model_path=str(MODEL_PATH))
    yield provider
    provider.unload()


async def test_transcribe_recovers_a_recognizable_word(stt):
    speech_16k = await _synthesize_16k_speech("The quick brown fox jumps over the lazy dog.")
    result = await stt.transcribe(speech_16k)
    assert "fox" in result.text.lower() or "dog" in result.text.lower(), (
        f"transcription didn't contain an expected word: {result.text!r}"
    )


async def test_transcribe_returns_empty_text_for_silence(stt):
    silence = b"\x00\x00" * 16000  # 1 second of silence at 16kHz
    result = await stt.transcribe(silence)
    assert result.text.strip() == "" or len(result.text.strip()) < 5


async def test_transcribe_passes_through_requested_language(stt):
    speech_16k = await _synthesize_16k_speech("Hello world.")
    result = await stt.transcribe(speech_16k, language="en")
    assert result.language == "en"


def test_known_non_speech_tag_is_stripped():
    from comfyui_realtime.providers.whisper_cpp_stt import _strip_known_non_speech_tags

    assert _strip_known_non_speech_tags("[BLANK_AUDIO]") == ""
    assert _strip_known_non_speech_tags("[MUSIC]") == ""


def test_unknown_bracketed_text_is_preserved():
    from comfyui_realtime.providers.whisper_cpp_stt import _strip_known_non_speech_tags

    assert _strip_known_non_speech_tags("see section [APPENDIX B]") == "see section [APPENDIX B]"
    assert _strip_known_non_speech_tags("the agency is called [NASA]") == "the agency is called [NASA]"


def test_known_tag_embedded_mid_sentence_collapses_whitespace():
    from comfyui_realtime.providers.whisper_cpp_stt import _strip_known_non_speech_tags

    assert _strip_known_non_speech_tags("hello [MUSIC] world") == "hello world"
