from comfyui_realtime.providers.base import (
    ChatMessage,
    GenerationDelta,
    GenerationOptions,
    VoiceInfo,
)
from comfyui_realtime.providers.base import TranscriptionResult, VADResult


def test_chat_message_fields():
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_generation_options_defaults():
    options = GenerationOptions()
    assert options.temperature == 0.8
    assert options.max_tokens is None


def test_generation_delta_fields():
    delta = GenerationDelta(text="hi", finished=False)
    assert delta.text == "hi"
    assert delta.finished is False


def test_voice_info_fields():
    voice = VoiceInfo(id="lessac-medium", name="Lessac (medium)")
    assert voice.id == "lessac-medium"
    assert voice.name == "Lessac (medium)"


def test_vad_result_defaults():
    result = VADResult(speech_probability=0.9)
    assert result.speech_probability == 0.9
    assert result.speech_started is False
    assert result.speech_ended is False


def test_vad_result_with_boundary_flags():
    result = VADResult(speech_probability=0.95, speech_started=True)
    assert result.speech_started is True
    assert result.speech_ended is False


def test_transcription_result_fields():
    result = TranscriptionResult(text="hello there", language="en")
    assert result.text == "hello there"
    assert result.language == "en"


def test_transcription_result_language_defaults_to_none():
    result = TranscriptionResult(text="hi")
    assert result.language is None
