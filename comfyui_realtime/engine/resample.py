"""Engine-owned audio resampling.

The wire format is fixed: pcm16, 24kHz, mono. Providers are not: Piper
voices emit 22.05kHz (medium/high) or 16kHz (low); Whisper/Silero need
16kHz. Providers declare their native rate and never resample themselves --
this module is the only place a sample rate conversion happens, so there is
exactly one place to get it right.

Uses soxr (not torchaudio): this module has no other dependency on torch or
ComfyUI, and soxr is small, fast, and not GPL. Does NOT use the stdlib
audioop -- removed in Python 3.13.
"""
from __future__ import annotations

import numpy as np
import soxr


def resample_pcm16(audio: bytes, from_rate: int, to_rate: int) -> bytes:
    if from_rate == to_rate:
        return audio
    samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
    resampled = soxr.resample(samples, from_rate, to_rate, quality="HQ")
    clipped = np.clip(resampled * 32768.0, -32768, 32767).astype("<i2")
    return clipped.tobytes()
