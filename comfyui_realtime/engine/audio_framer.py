# comfyui_realtime/engine/audio_framer.py
"""VAD frame re-chunking.

Client audio arrives over the wire in whatever chunk sizes the client's
input_audio_buffer.append calls happen to use -- they won't align to VAD's
required exact-512-sample (16kHz) frame boundary, hence the ring buffer.
This module does NOT resample -- the caller (the audio input loop)
resamples once and feeds the same 16kHz bytes here AND into its own
STT-utterance accumulator, rather than resampling twice for two consumers.
"""
from __future__ import annotations

FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2  # 16-bit pcm


class AudioFrameBuffer:
    def __init__(self) -> None:
        self._buffer = b""

    def push(self, audio_16k: bytes) -> list[bytes]:
        self._buffer += audio_16k
        frames = []
        while len(self._buffer) >= FRAME_BYTES:
            frames.append(self._buffer[:FRAME_BYTES])
            self._buffer = self._buffer[FRAME_BYTES:]
        return frames

    def flush(self) -> bytes | None:
        if not self._buffer:
            return None
        padding_needed = FRAME_BYTES - len(self._buffer)
        padded = self._buffer + b"\x00" * padding_needed
        self._buffer = b""
        return padded
