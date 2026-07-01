"""Full Phase 1 round trip: OpenAI SDK -> real LlamaCppLLMProvider ->
real PiperTTSProvider, against a live ComfyUI server with the
'phase1-tts' pipeline already registered (see scripts/register_phase1_pipeline.json).

SDK adaptation (same pattern as the Phase 0 echo-pipeline integration test):
uses `client.realtime.connect(...)`
(the GA namespace), not `client.beta.realtime.connect(...)`. The installed
SDK is openai==2.43.0, whose `client.beta.realtime` hardcodes a `wss://`
URL and always sends `OpenAI-Beta: realtime=v1` regardless of the client's
base_url scheme -- it fails outright against this plain-http local dev
server. `client.realtime` derives ws/wss from base_url's own scheme and
sends no beta header, the correct match for a GA-dialect session per spec
section 6.2.
"""
import base64
import os

import pytest
from openai import AsyncOpenAI

from .conftest import COMFYUI_URL

pytestmark = pytest.mark.integration


async def test_full_text_to_speech_round_trip():
    client = AsyncOpenAI(api_key="not-needed", base_url=f"{COMFYUI_URL}/v1")
    async with client.realtime.connect(model="phase1-tts") as connection:
        await connection.conversation.item.create(
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hello."}]}
        )
        await connection.response.create()

        text_deltas = []
        audio_byte_count = 0
        saw_response_done = False
        async for event in connection:
            if event.type == "response.output_text.delta":
                text_deltas.append(event.delta)
            elif event.type == "response.output_audio.delta":
                audio_byte_count += len(base64.b64decode(event.delta))
            elif event.type == "response.done":
                saw_response_done = True
                break

        full_text = "".join(text_deltas)
        assert len(full_text.strip()) > 0, "LLM produced no text"
        assert audio_byte_count > 0, "TTS produced no audio bytes"
        assert saw_response_done
