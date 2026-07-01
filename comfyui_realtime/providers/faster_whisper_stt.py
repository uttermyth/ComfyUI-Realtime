"""
Wraps faster-whisper -- a maintained whisper.cpp binding (faster-whisper is a Python wrapper around whisper.cpp).
"""

import asyncio
import threading

import numpy as np
from faster_whisper import WhisperModel

from .base import TranscriptionResult

class FasterWhisperSTTProvider:
    sample_rate = 16000

    def __init__(self, model_path: str) -> None:
        self._model = WhisperModel(model_path)
        self._lock = threading.Lock()

    async def transcribe(self, audio_buffer: bytes, language: str | None = None) -> TranscriptionResult:
        return await asyncio.to_thread(self._transcribe_sync, audio_buffer, language)

    def _transcribe_sync(self, audio_buffer: bytes, language: str | None) -> TranscriptionResult:
        audio = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            segments, _ = self._model.transcribe(audio, language=language)
            print(f"Faster Whisper STT Result: {segments}")
            text = " ".join(segment.text for segment in segments)
            return TranscriptionResult(text=text, language=language)
        
    def unload(self) -> None:
        del self._model