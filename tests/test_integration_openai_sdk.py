"""Requires a live ComfyUI server with the echo pipeline already registered
(see scripts/register_echo_pipeline.json). Run with: pytest tests/test_integration_openai_sdk.py -m integration
This substitutes for a Uttermyth-specific WebSocketConnectionManager check,
since Uttermyth's source isn't available in this workspace -- the official
OpenAI SDK's realtime client still exercises the same compatibility surface.

Adaptations made against the OpenAI Realtime API's beta sketch:

1. `client.realtime.connect(...)` (the GA namespace), not
   `client.beta.realtime.connect(...)`. The installed SDK is openai 2.43.0,
   whose `client.beta.realtime` always dials `wss://` regardless of the
   client's base_url scheme (and always sends `OpenAI-Beta: realtime=v1`,
   which our server's dialects.select_dialect() accepts only as a
   GA-fallback-with-warning, per spec section 6.2 Phase 0 scope). The
   non-beta `client.realtime.connect(...)` derives ws/wss from base_url's
   own scheme and sends no beta header -- the correct match for a plain-http
   local dev server speaking GA-only Phase 0 wire format.

2. The expected echoed text is "You said: ", not "echo: hello". The
   brief's own scripts/register_echo_pipeline.json registers
   EchoLLMProviderNode with fixed_response="You said: ", and
   EchoLLMProviderNode (Task 6, not modified by this task) always returns
   that fixed string regardless of input -- it never echoes the input text.
   "echo: hello" was only ever true of tests/test_websocket_handler.py's
   local _StubLLM, which is unrelated to the real registered pipeline.
"""
import pytest
from openai import AsyncOpenAI

from .conftest import COMFYUI_URL

pytestmark = pytest.mark.integration


async def test_openai_sdk_text_round_trip_against_echo_pipeline():
    client = AsyncOpenAI(api_key="not-needed", base_url=f"{COMFYUI_URL}/v1")
    async with client.realtime.connect(model="echo") as connection:
        await connection.conversation.item.create(
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
        )
        deltas = []
        async for event in connection:
            if event.type == "response.output_text.delta":
                deltas.append(event.delta)
            elif event.type == "response.done":
                break
        assert "".join(deltas) == "You said: "
