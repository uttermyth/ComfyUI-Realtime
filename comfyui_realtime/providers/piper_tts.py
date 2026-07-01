"""PiperTTSProvider wraps piper-tts (OHF-Voice/piper1-gpl, GPL-3.0 --
kept as an optional extra so this package's own license is unaffected).

Each text chunk handed to synthesize() gets its own
engine/executor_bridge.py call (one worker thread per sentence, not one
long-lived thread for the whole response) -- simpler than a persistent
bridge across the whole stream, and thread-spawn overhead for a handful of
sentences per response is negligible next to synthesis time itself.

Piper voices emit their own native rate (22.05kHz for medium/high voices,
16kHz for low) -- this provider never emits that rate directly; every
chunk is resampled to the wire rate via engine/resample.py before being
yielded.
"""
from __future__ import annotations

import threading
from typing import AsyncIterator

from piper import PiperVoice

from ..engine.executor_bridge import bridge_sync_iterator
from ..engine.resample import resample_pcm16
from .base import VoiceInfo


class PiperTTSProvider:
    output_format = "pcm16"

    def __init__(self, voices: dict[str, tuple[str, str]], default_voice: str) -> None:
        self._voices: dict[str, PiperVoice] = {
            voice_id: PiperVoice.load(onnx_path, config_path=config_path)
            for voice_id, (onnx_path, config_path) in voices.items()
        }
        self._default_voice = default_voice
        self._lock = threading.Lock()
        self.output_sample_rate = 24000

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id=voice_id, name=voice_id) for voice_id in self._voices]

    async def synthesize(
        self, text_stream: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[bytes]:
        piper_voice = self._voices[voice or self._default_voice]
        native_rate = piper_voice.config.sample_rate

        self._lock.acquire()
        try:
            async for text in text_stream:
                stop_event = threading.Event()

                def factory(text: str = text):
                    for chunk in piper_voice.synthesize(text):
                        yield chunk.audio_int16_bytes

                bridge = bridge_sync_iterator(factory, stop_event)
                try:
                    async for raw_chunk in bridge:
                        yield resample_pcm16(raw_chunk, native_rate, self.output_sample_rate)
                finally:
                    # Explicitly await this chunk's bridge.aclose() so its
                    # worker-thread join completes before we move to the next
                    # sentence or let the outer lock release -- see
                    # LlamaCppLLMProvider's docstring for the full mechanism.
                    # Without this, aclose() on `synthesize()` would release
                    # `self._lock` while this chunk's worker thread is still
                    # mid-decode, letting a second synthesize() call race it
                    # on Piper's underlying ONNX session.
                    await bridge.aclose()
        finally:
            self._lock.release()

    def unload(self) -> None:
        self._voices.clear()
