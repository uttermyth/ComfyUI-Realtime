"""Full Phase 2 round trip: real synthesized speech (via Piper, at the
wire's 24kHz pcm16 format) sent as input_audio_buffer.append chunks of
deliberately arbitrary, non-frame-aligned sizes -- against a live ComfyUI
server with the 'phase2-full' pipeline (real VAD + STT + LLM + TTS)
already registered (see Task 11, Steps 1-2). No Uttermyth substitution
needed -- this is spec section 10's own Phase 2 success criterion: an
unmodified OpenAI Realtime SDK client working with only the URL changed.
"""
import asyncio
import base64
import os
import pathlib

import pytest
from openai import AsyncOpenAI

pytestmark = pytest.mark.integration

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
VOICES_DIR = pathlib.Path(__file__).parent.parent / "assets" / "piper_voices" / "en_US" / "lessac"


async def _synthesize_24k_wire_audio(text: str) -> bytes:
    from comfyui_realtime.providers.piper_tts import PiperTTSProvider

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
        return b"".join(chunks)  # PiperTTSProvider already resamples to the 24kHz wire rate
    finally:
        provider.unload()


async def test_full_speech_to_speech_round_trip():
    speech_24k = await _synthesize_24k_wire_audio("What is the capital of France?")
    trailing_silence = b"\x00\x00" * 24000  # 1s of silence so VAD reliably detects speech_ended

    client = AsyncOpenAI(api_key="not-needed", base_url=f"{COMFYUI_URL}/v1")
    async with client.realtime.connect(model="phase2-full") as connection:
        # Deliberately arbitrary, non-frame-aligned chunk size -- exercises
        # the real ring-buffer/resampling path end to end, not a
        # convenient round number.
        full_audio = speech_24k + trailing_silence
        chunk_size = 4001
        for i in range(0, len(full_audio), chunk_size):
            chunk = full_audio[i : i + chunk_size]
            await connection.input_audio_buffer.append(audio=base64.b64encode(chunk).decode("ascii"))

        transcript_text = ""
        response_text_deltas = []
        audio_byte_count = 0
        saw_speech_started = False
        saw_response_done = False

        async for event in connection:
            if event.type == "input_audio_buffer.speech_started":
                saw_speech_started = True
            elif event.type == "conversation.item.input_audio_transcription.completed":
                transcript_text = event.transcript
            elif event.type == "response.output_text.delta":
                response_text_deltas.append(event.delta)
            elif event.type == "response.output_audio.delta":
                audio_byte_count += len(base64.b64decode(event.delta))
            elif event.type == "response.done":
                saw_response_done = True
                break

        print(f"\ntranscript: {transcript_text!r}")
        print(f"response text: {''.join(response_text_deltas)!r}")
        print(f"response audio bytes: {audio_byte_count}")

        assert saw_speech_started, "VAD never detected speech in the synthesized audio"
        assert len(transcript_text.strip()) > 0, "STT produced no transcript"
        assert len("".join(response_text_deltas).strip()) > 0, "LLM produced no response text"
        assert audio_byte_count > 0, "TTS produced no response audio"
        assert saw_response_done
