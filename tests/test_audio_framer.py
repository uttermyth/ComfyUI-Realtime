# tests/test_audio_framer.py
import math
import struct

from comfyui_realtime.engine.audio_framer import FRAME_BYTES, AudioFrameBuffer


def _make_sine_pcm16(duration_s: float, rate: int, freq: float = 220.0) -> bytes:
    n_samples = int(duration_s * rate)
    samples = [int(20000 * math.sin(2 * math.pi * freq * (i / rate))) for i in range(n_samples)]
    return struct.pack(f"<{n_samples}h", *samples)


def test_push_returns_no_frames_for_audio_shorter_than_one_frame():
    buf = AudioFrameBuffer()
    tiny = _make_sine_pcm16(0.005, 16000)
    frames = buf.push(tiny)
    assert frames == []


def test_push_returns_exact_512_sample_frames():
    buf = AudioFrameBuffer()
    one_second = _make_sine_pcm16(1.0, 16000)
    frames = buf.push(one_second)
    assert len(frames) > 0
    for frame in frames:
        assert len(frame) == FRAME_BYTES


def test_push_accumulates_remainder_across_calls():
    buf = AudioFrameBuffer()
    half_second = _make_sine_pcm16(0.5, 16000)
    frames1 = buf.push(half_second)
    frames2 = buf.push(half_second)
    all_frames = frames1 + frames2
    assert len(all_frames) > 0
    for frame in all_frames:
        assert len(frame) == FRAME_BYTES


def test_flush_on_fresh_buffer_returns_none():
    buf = AudioFrameBuffer()
    assert buf.flush() is None


def test_flush_zero_pads_a_partial_frame_to_exact_frame_size():
    buf = AudioFrameBuffer()
    buf.push(_make_sine_pcm16(0.001, 16000))  # guaranteed less than one full 512-sample frame
    remainder = buf.flush()
    assert remainder is not None
    assert len(remainder) == FRAME_BYTES


def test_flush_then_push_starts_clean():
    buf = AudioFrameBuffer()
    buf.push(_make_sine_pcm16(0.001, 16000))
    buf.flush()
    assert buf.flush() is None
