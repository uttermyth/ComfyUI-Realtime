"""WhisperCppSTTProvider wraps pywhispercpp -- the maintained whisper.cpp
binding (whisper-cpp-python is stale, do not use). transcribe() is
one-shot on a complete buffered utterance (streaming STT is not
implemented) and dispatched via asyncio.to_thread() -- a single blocking
call, not a stream (see Global Constraints for why this differs from the
LLM/TTS executor_bridge pattern).

Non-speech audio (silence, music, etc.) decodes to a literal bracketed tag
like "[BLANK_AUDIO]" rather than an empty segment list -- a whisper.cpp/
ggml model convention, not a pywhispercpp quirk.
Those tags are stripped here so callers see a genuinely empty string for
non-speech. Stripping is anchored to a known vocabulary of whisper.cpp's
non-speech-event tags (not a blanket "any bracketed caps text" strip) so
genuine spoken content that happens to render bracketed -- an acronym, a
read-aloud section reference -- is never silently dropped.
"""
from __future__ import annotations

import asyncio
import re
import threading

import numpy as np
from pywhispercpp.model import Model

from .base import TranscriptionResult

# Known whisper.cpp/Whisper non-speech-event tags -- intentionally a
# vocabulary allowlist, not "any bracketed caps text", so genuine spoken
# content that happens to render bracketed (an acronym, a read-aloud
# section reference) is never silently stripped.
_NON_SPEECH_TAGS = frozenset(
    {"BLANK_AUDIO", "SILENCE", "MUSIC", "NOISE", "LAUGHTER", "APPLAUSE", "INAUDIBLE"}
)
_BRACKETED_TAG_RE = re.compile(r"\[([A-Z _]+)\]")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_known_non_speech_tags(text: str) -> str:
    def _replace(match: "re.Match[str]") -> str:
        return "" if match.group(1).strip() in _NON_SPEECH_TAGS else match.group(0)

    stripped = _BRACKETED_TAG_RE.sub(_replace, text)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


class WhisperCppSTTProvider:
    sample_rate = 16000

    def __init__(self, model_path: str) -> None:
        self._model = Model(model_path)
        self._lock = threading.Lock()

    async def transcribe(self, audio_buffer: bytes, language: str | None = None) -> TranscriptionResult:
        return await asyncio.to_thread(self._transcribe_sync, audio_buffer, language)

    def _transcribe_sync(self, audio_buffer: bytes, language: str | None) -> TranscriptionResult:
        samples = np.frombuffer(audio_buffer, dtype="<i2").astype(np.float32) / 32768.0
        with self._lock:
            kwargs = {"language": language} if language else {}
            segments = self._model.transcribe(samples, **kwargs)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        text = _strip_known_non_speech_tags(text)
        return TranscriptionResult(text=text, language=language)

    def unload(self) -> None:
        del self._model
