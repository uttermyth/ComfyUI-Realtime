import math
import struct

from comfyui_realtime.engine.resample import resample_pcm16


def _make_sine_pcm16(duration_s: float, rate: int, freq: float = 220.0) -> bytes:
    n_samples = int(duration_s * rate)
    samples = [int(20000 * math.sin(2 * math.pi * freq * (i / rate))) for i in range(n_samples)]
    return struct.pack(f"<{n_samples}h", *samples)


def test_resample_changes_sample_count_proportionally():
    pcm_16k = _make_sine_pcm16(1.0, 16000)
    resampled = resample_pcm16(pcm_16k, from_rate=16000, to_rate=24000)
    samples_in = len(pcm_16k) // 2
    samples_out = len(resampled) // 2
    # Allow a small tolerance for resampler edge effects.
    assert abs(samples_out - samples_in * 24000 / 16000) < 50


def test_resample_22050_to_24000():
    pcm_22050 = _make_sine_pcm16(1.0, 22050)
    resampled = resample_pcm16(pcm_22050, from_rate=22050, to_rate=24000)
    samples_in = len(pcm_22050) // 2
    samples_out = len(resampled) // 2
    assert abs(samples_out - samples_in * 24000 / 22050) < 50


def test_resample_to_same_rate_is_a_passthrough():
    pcm = _make_sine_pcm16(0.5, 24000)
    resampled = resample_pcm16(pcm, from_rate=24000, to_rate=24000)
    assert resampled == pcm


def test_output_is_still_valid_pcm16_bytes():
    pcm_16k = _make_sine_pcm16(0.2, 16000)
    resampled = resample_pcm16(pcm_16k, from_rate=16000, to_rate=24000)
    assert len(resampled) % 2 == 0  # whole number of 16-bit samples
