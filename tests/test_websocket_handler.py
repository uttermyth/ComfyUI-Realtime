import asyncio
import base64
import json

import pytest
from aiohttp import web

from comfyui_realtime.nodes.pipeline_node import EchoLLMProviderNode
from comfyui_realtime.providers.base import GenerationDelta, TranscriptionResult, VADResult
from comfyui_realtime.registry import PipelineConfig, pipeline_registry
from comfyui_realtime.server.websocket_handler import realtime_websocket_handler


class _StubLLM:
    async def generate(self, messages, options):
        text = messages[-1].content if messages else ""
        yield GenerationDelta(text=f"echo: {text}", finished=False)
        yield GenerationDelta(text="", finished=True)


class _StubTTS:
    output_sample_rate = 24000
    output_format = "pcm16"

    async def synthesize(self, text_stream, voice=None):
        async for text in text_stream:
            yield f"audio-for:{text}".encode("ascii")

    def list_voices(self):
        return []


async def _build_app():
    app = web.Application()
    app.router.add_get("/v1/realtime", realtime_websocket_handler)
    return app


async def test_pipeline_not_found_sends_error_and_closes(aiohttp_client):
    app = await _build_app()
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/v1/realtime?model=nonexistent")
    msg = await ws.receive()
    payload = json.loads(msg.data)
    assert payload["type"] == "error"
    assert payload["error"]["code"] == "pipeline_not_found"
    closed = await ws.receive()
    assert closed.type.name == "CLOSE"


async def test_session_registered_before_pipeline_lookup_closing_a_toctou_race(aiohttp_client, monkeypatch):
    """Regression test for a race found during Task 3's safety review: the
    conservative unload rule checks session_registry.count() to decide
    whether it's safe to unload a provider. If a new connection looked up
    its pipeline (capturing a provider reference) BEFORE registering
    itself as an active session, a concurrent pipeline deletion landing in
    that exact window would see count() == 0 and unload a provider this
    connection was about to use. This proves the ordering is the other way
    around: the session is already counted by the time the lookup runs."""
    from comfyui_realtime import registry
    from comfyui_realtime.session_registry import session_registry

    observed_counts_at_lookup_time = []
    real_get = registry.pipeline_registry.get

    def _spying_get(name):
        observed_counts_at_lookup_time.append(session_registry.count())
        return real_get(name)

    monkeypatch.setattr(registry.pipeline_registry, "get", _spying_get)

    pipeline_registry.register(PipelineConfig(name="race-order-test", llm=_StubLLM()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        before = session_registry.count()
        ws = await client.ws_connect("/v1/realtime?model=race-order-test")
        await ws.receive()  # session.created
        assert len(observed_counts_at_lookup_time) == 1
        assert observed_counts_at_lookup_time[0] == before + 1, (
            "the connecting session must already be counted by session_registry "
            "at the moment the pipeline lookup runs"
        )
    finally:
        pipeline_registry.unregister("race-order-test")


async def test_item_create_appends_history_and_acks_without_generating(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="echo", llm=_StubLLM()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=echo")
        await ws.receive()  # session.created

        await ws.send_json(
            {
                "type": "conversation.item.create",
                "item": {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            }
        )

        added = json.loads((await ws.receive()).data)
        assert added["type"] == "conversation.item.added"
        assert added["item"]["content"][0]["text"] == "hello"

        done = json.loads((await ws.receive()).data)
        assert done["type"] == "conversation.item.done"

        # No response.* events should follow -- item.create alone never
        # triggers generation (spec Appendix A; corrects Phase 0's
        # auto-trigger behavior).
        await ws.send_json({"type": "ping_marker"})  # unhandled type, ignored
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ws.receive(), timeout=0.2)
    finally:
        pipeline_registry.unregister("echo")


async def test_multiple_item_creates_accumulate_history(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="echo", llm=_StubLLM()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=echo")
        await ws.receive()  # session.created

        for text in ("first", "second"):
            await ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {"role": "user", "content": [{"type": "input_text", "text": text}]},
                }
            )
            await ws.receive()  # .added
            await ws.receive()  # .done

        # Both items landed in history -- verified indirectly in Task 9's
        # response.create test, which reads this same history list.
    finally:
        pipeline_registry.unregister("echo")


async def test_response_create_text_only_pipeline(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="echo", llm=_StubLLM()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=echo")
        await ws.receive()  # session.created

        await ws.send_json(
            {
                "type": "conversation.item.create",
                "item": {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            }
        )
        await ws.receive()  # item.added
        await ws.receive()  # item.done

        await ws.send_json({"type": "response.create"})

        created = json.loads((await ws.receive()).data)
        assert created["type"] == "response.created"

        delta = json.loads((await ws.receive()).data)
        assert delta["type"] == "response.output_text.delta"
        assert delta["delta"] == "echo: hello"

        text_done = json.loads((await ws.receive()).data)
        assert text_done["type"] == "response.output_text.done"

        response_done = json.loads((await ws.receive()).data)
        assert response_done["type"] == "response.done"
    finally:
        pipeline_registry.unregister("echo")


async def test_response_create_with_real_echo_node_provider(aiohttp_client):
    # Regression test for the C1 bug (final whole-branch review): every other
    # response.create test here drives a hand-written _StubLLM, never the
    # actual EchoLLMProviderNode that ships in NODE_CLASS_MAPPINGS and
    # scripts/register_echo_pipeline.json. That gap let _FixedStringLLM's
    # stale Phase 0 generate(self, text) signature ship broken against the
    # real ILLMProvider streaming interface. This test builds the LLM via
    # the real node so it would have caught that TypeError directly.
    (llm,) = EchoLLMProviderNode.execute(fixed_response="You said: ").result
    pipeline_registry.register(PipelineConfig(name="echo-real-node", llm=llm))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=echo-real-node")
        await ws.receive()  # session.created

        await ws.send_json(
            {
                "type": "conversation.item.create",
                "item": {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            }
        )
        await ws.receive()  # item.added
        await ws.receive()  # item.done

        await ws.send_json({"type": "response.create"})

        created = json.loads((await ws.receive()).data)
        assert created["type"] == "response.created"

        delta = json.loads((await ws.receive()).data)
        assert delta["type"] == "response.output_text.delta"
        assert delta["delta"] == "You said: "

        text_done = json.loads((await ws.receive()).data)
        assert text_done["type"] == "response.output_text.done"

        response_done = json.loads((await ws.receive()).data)
        assert response_done["type"] == "response.done"
    finally:
        pipeline_registry.unregister("echo-real-node")


class _SlowStubLLM:
    """Takes real async time to generate, so tests can prove the main
    message loop stays responsive while a response is in flight."""

    async def generate(self, messages, options):
        await asyncio.sleep(0.3)
        yield GenerationDelta(text="slow response", finished=False)
        yield GenerationDelta(text="", finished=True)


async def test_response_create_does_not_block_the_message_loop(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="slow", llm=_SlowStubLLM()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=slow")
        await ws.receive()  # session.created

        await ws.send_json(
            {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": "go"}]}}
        )
        await ws.receive()  # item.added
        await ws.receive()  # item.done

        await ws.send_json({"type": "response.create"})
        created = json.loads((await ws.receive()).data)
        assert created["type"] == "response.created"

        # While the slow response is still generating (it sleeps 0.3s
        # before yielding anything), the message loop must still accept
        # and ack a second item.create -- this would hang if
        # response.create blocked the loop the way Phase 1's directly-
        # awaited version did.
        await ws.send_json(
            {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": "while busy"}]}}
        )
        added = await asyncio.wait_for(ws.receive(), timeout=1.0)
        assert json.loads(added.data)["type"] == "conversation.item.added"
    finally:
        pipeline_registry.unregister("slow")


async def test_second_response_create_while_one_active_is_ignored(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="slow2", llm=_SlowStubLLM()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=slow2")
        await ws.receive()

        await ws.send_json(
            {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": "go"}]}}
        )
        await ws.receive()
        await ws.receive()

        await ws.send_json({"type": "response.create"})
        await ws.receive()  # first response's response.created

        await ws.send_json({"type": "response.create"})  # must be ignored
        next_event = json.loads((await ws.receive()).data)
        assert next_event["type"] != "response.created"
    finally:
        pipeline_registry.unregister("slow2")


async def test_response_create_with_tts_emits_audio_events(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="echo-tts", llm=_StubLLM(), tts=_StubTTS()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=echo-tts")

        created_session = json.loads((await ws.receive()).data)
        assert created_session["session"]["modalities"]["output"] == ["text", "audio"]

        await ws.send_json(
            {
                "type": "conversation.item.create",
                "item": {"role": "user", "content": [{"type": "input_text", "text": "Hi there."}]},
            }
        )
        await ws.receive()  # item.added
        await ws.receive()  # item.done

        await ws.send_json({"type": "response.create"})

        events_received = []
        while True:
            msg = json.loads((await ws.receive()).data)
            events_received.append(msg["type"])
            if msg["type"] == "response.done":
                break

        assert "response.created" in events_received
        assert "response.output_text.delta" in events_received
        assert "response.output_text.done" in events_received
        assert "response.output_audio.delta" in events_received
        assert "response.output_audio.done" in events_received
        assert "response.output_audio_transcript.delta" in events_received
        assert "response.output_audio_transcript.done" in events_received
        assert events_received[-1] == "response.done"
    finally:
        pipeline_registry.unregister("echo-tts")


class _StubVAD:
    """Deterministic VAD stand-in: fires speech_started on the Nth frame,
    speech_ended some frames later, regardless of actual audio content --
    keeps this test independent of Silero's real model behavior (covered
    by Task 3's own tests)."""

    def __init__(self, start_at_frame: int, end_at_frame: int):
        self._start_at = start_at_frame
        self._end_at = end_at_frame
        self._count = 0

    async def analyze(self, audio_chunk):
        self._count += 1
        return VADResult(
            speech_probability=0.9,
            speech_started=(self._count == self._start_at),
            speech_ended=(self._count == self._end_at),
        )


class _StubSTT:
    sample_rate = 16000

    def __init__(self, fixed_text: str):
        self._fixed_text = fixed_text

    async def transcribe(self, audio_buffer, language=None):
        return TranscriptionResult(text=self._fixed_text)


def _silence_frame() -> bytes:
    # 24kHz wire-rate silence, sized so one append produces multiple
    # resampled 16kHz frames once the handler resamples it.
    return b"\x00\x00" * 1536


async def test_audio_input_loop_detects_turn_and_triggers_response(aiohttp_client):
    vad = _StubVAD(start_at_frame=2, end_at_frame=5)
    stt = _StubSTT(fixed_text="hello from audio")
    pipeline_registry.register(PipelineConfig(name="audio-in", llm=_StubLLM(), vad=vad, stt=stt))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=audio-in")
        session_created = json.loads((await ws.receive()).data)
        assert session_created["session"]["modalities"]["input"] == ["text", "audio"]

        for _ in range(8):
            await ws.send_json(
                {"type": "input_audio_buffer.append", "audio": base64.b64encode(_silence_frame()).decode("ascii")}
            )

        events_received = []
        while True:
            msg = json.loads((await ws.receive()).data)
            events_received.append(msg["type"])
            if msg["type"] == "response.done":
                break

        assert "input_audio_buffer.speech_started" in events_received
        assert "input_audio_buffer.speech_stopped" in events_received
        assert "input_audio_buffer.committed" in events_received
        assert "conversation.item.input_audio_transcription.completed" in events_received
        assert "response.created" in events_received
        assert events_received[-1] == "response.done"
    finally:
        pipeline_registry.unregister("audio-in")


async def test_audio_input_loop_handles_non_frame_aligned_append_chunks(aiohttp_client):
    """Regression test (Task 11): input_audio_buffer.append chunk sizes are
    client-controlled and not guaranteed to be a whole number of pcm16
    samples (2 bytes each). A live end-to-end run with chunk_size=4001
    (odd) crashed resample_pcm16 with "buffer size must be a multiple of
    element size" because the raw wire chunk was resampled directly,
    before any byte-alignment buffering happened. The fix carries a
    trailing odd byte over to the next append call via
    SessionState.pending_wire_audio."""
    vad = _StubVAD(start_at_frame=2, end_at_frame=5)
    stt = _StubSTT(fixed_text="hello from audio")
    pipeline_registry.register(PipelineConfig(name="audio-in-odd", llm=_StubLLM(), vad=vad, stt=stt))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=audio-in-odd")
        await ws.receive()  # session.created

        full_audio = _silence_frame() * 8
        chunk_size = 4001  # deliberately odd and not frame-aligned
        for i in range(0, len(full_audio), chunk_size):
            chunk = full_audio[i : i + chunk_size]
            await ws.send_json(
                {"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")}
            )

        events_received = []
        while True:
            msg = json.loads((await ws.receive()).data)
            events_received.append(msg["type"])
            if msg["type"] == "response.done":
                break

        assert "input_audio_buffer.speech_started" in events_received
        assert "input_audio_buffer.speech_stopped" in events_received
        assert events_received[-1] == "response.done"
    finally:
        pipeline_registry.unregister("audio-in-odd")


async def test_append_without_stt_returns_protocol_error(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="no-audio", llm=_StubLLM()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=no-audio")
        await ws.receive()  # session.created

        await ws.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(_silence_frame()).decode("ascii")}
        )
        error = json.loads((await ws.receive()).data)
        assert error["type"] == "error"
        assert error["error"]["code"] == "audio_input_not_supported"
    finally:
        pipeline_registry.unregister("no-audio")


class _SlowStubLLMForCancel:
    """Yields multiple deltas slowly, giving a test time to trigger
    cancellation mid-generation."""

    async def generate(self, messages, options):
        for word in ["one ", "two ", "three ", "four ", "five "]:
            await asyncio.sleep(0.1)
            yield GenerationDelta(text=word, finished=False)
        yield GenerationDelta(text="", finished=True)


async def test_vad_speech_started_cancels_active_response(aiohttp_client):
    vad = _StubVAD(start_at_frame=1, end_at_frame=100)  # never naturally ends in this test
    stt = _StubSTT(fixed_text="ignored")
    pipeline_registry.register(PipelineConfig(name="barge-in", llm=_SlowStubLLMForCancel(), vad=vad, stt=stt))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=barge-in")
        await ws.receive()  # session.created

        await ws.send_json(
            {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": "go"}]}}
        )
        await ws.receive()
        await ws.receive()

        await ws.send_json({"type": "response.create"})
        await ws.receive()  # response.created
        await ws.receive()  # first output_text.delta ("one ")

        await ws.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(_silence_frame()).decode("ascii")}
        )

        events_received = []
        for _ in range(6):
            msg = json.loads((await ws.receive()).data)
            events_received.append(msg)
            if msg["type"] == "input_audio_buffer.speech_started":
                break

        done_events = [e for e in events_received if e["type"] == "response.done"]
        assert len(done_events) == 1
        assert done_events[0]["response"]["status"] == "cancelled"
        assert events_received[-1]["type"] == "input_audio_buffer.speech_started"
    finally:
        pipeline_registry.unregister("barge-in")


async def test_explicit_response_cancel(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="cancel-explicit", llm=_SlowStubLLMForCancel()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=cancel-explicit")
        await ws.receive()

        await ws.send_json(
            {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": "go"}]}}
        )
        await ws.receive()
        await ws.receive()

        await ws.send_json({"type": "response.create"})
        await ws.receive()  # response.created
        await ws.receive()  # first delta

        await ws.send_json({"type": "response.cancel"})

        msg = json.loads((await ws.receive()).data)
        assert msg["type"] == "response.done"
        assert msg["response"]["status"] == "cancelled"
    finally:
        pipeline_registry.unregister("cancel-explicit")


async def test_conversation_item_truncate_does_not_crash_on_unmatched_item(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="truncate-test", llm=_SlowStubLLMForCancel()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=truncate-test")
        await ws.receive()

        await ws.send_json(
            {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": "go"}]}}
        )
        await ws.receive()
        await ws.receive()

        await ws.send_json({"type": "response.create"})
        await ws.receive()  # response.created
        await ws.receive()  # a delta
        await ws.send_json({"type": "response.cancel"})
        while True:
            msg = json.loads((await ws.receive()).data)
            if msg["type"] == "response.done":
                break

        await ws.send_json({"type": "conversation.item.truncate", "item_id": "irrelevant", "audio_end_ms": 0})
        truncated = json.loads((await ws.receive()).data)
        assert truncated["type"] == "conversation.item.truncated"
    finally:
        pipeline_registry.unregister("truncate-test")


async def test_commit_based_turn_detection_when_no_vad(aiohttp_client):
    stt = _StubSTT(fixed_text="manual turn text")
    pipeline_registry.register(PipelineConfig(name="manual-turn", llm=_StubLLM(), stt=stt))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=manual-turn")
        await ws.receive()  # session.created

        await ws.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(_silence_frame()).decode("ascii")}
        )
        await ws.send_json({"type": "input_audio_buffer.commit"})

        events_received = []
        while True:
            msg = json.loads((await ws.receive()).data)
            events_received.append(msg["type"])
            if msg["type"] == "response.done":
                break

        # No VAD -- speech_started/speech_stopped must never appear.
        assert "input_audio_buffer.speech_started" not in events_received
        assert "input_audio_buffer.committed" in events_received
        assert "conversation.item.input_audio_transcription.completed" in events_received
        assert events_received[-1] == "response.done"
    finally:
        pipeline_registry.unregister("manual-turn")


async def test_response_create_without_llm_returns_protocol_error(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="no-llm-test"))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=no-llm-test")
        await ws.receive()  # session.created

        await ws.send_json({"type": "response.create"})
        error = json.loads((await ws.receive()).data)
        assert error["type"] == "error"
        assert error["error"]["code"] == "llm_not_configured"
    finally:
        pipeline_registry.unregister("no-llm-test")


async def test_continuous_transcription_auto_trigger_is_silently_skipped(aiohttp_client):
    vad = _StubVAD(start_at_frame=1, end_at_frame=2)
    stt = _StubSTT(fixed_text="just transcribing, no response expected")
    pipeline_registry.register(PipelineConfig(name="transcription-only-skip-test", vad=vad, stt=stt))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=transcription-only-skip-test")
        await ws.receive()  # session.created

        await ws.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(_silence_frame()).decode("ascii")}
        )

        events_received = []
        for _ in range(6):
            msg = json.loads((await ws.receive()).data)
            events_received.append(msg["type"])
            if msg["type"] == "conversation.item.input_audio_transcription.completed":
                break

        assert "conversation.item.input_audio_transcription.completed" in events_received
        # No error either -- this is expected behavior, not a client mistake.
        assert not any(t == "error" or t.startswith("response.") for t in events_received)

        # The loop above breaks AT transcription.completed, so it would never
        # observe a leaked response.created sent right after -- without the
        # guard, _start_response_task would still fire here (its first action
        # is sending response.created, before it ever touches pipeline.llm).
        # Confirm nothing more arrives at all within a generous window.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ws.receive(), timeout=0.2)
    finally:
        pipeline_registry.unregister("transcription-only-skip-test")


async def test_session_update_is_acknowledged_without_being_applied(aiohttp_client):
    vad = _StubVAD(start_at_frame=100, end_at_frame=200)  # never fires
    stt = _StubSTT(fixed_text="unused")
    pipeline_registry.register(PipelineConfig(name="session-update-test", vad=vad, stt=stt))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=session-update-test")
        created = json.loads((await ws.receive()).data)

        # Try to change voice/instructions/temperature -- none of this
        # should be applied (spec section 1.4: live reconfiguration is a
        # non-goal). The ack must echo the REAL session config.
        await ws.send_json(
            {"type": "session.update", "session": {"voice": "alloy", "instructions": "ignored", "temperature": 1.9}}
        )
        updated = json.loads((await ws.receive()).data)

        assert updated["type"] == "session.updated"
        assert updated["session"]["id"] == created["session"]["id"]
        assert updated["session"]["turn_detection"] == created["session"]["turn_detection"]
        assert updated["session"]["modalities"] == created["session"]["modalities"]

        # Confirm nothing was actually applied to the pipeline's real config.
        config = pipeline_registry.get("session-update-test")
        assert config.voice is None
        assert config.instructions is None
        assert config.temperature == 0.8
    finally:
        pipeline_registry.unregister("session-update-test")


async def test_session_created_reports_server_vad_when_vad_present(aiohttp_client):
    vad = _StubVAD(start_at_frame=100, end_at_frame=200)  # never fires in this test
    stt = _StubSTT(fixed_text="unused")
    pipeline_registry.register(PipelineConfig(name="turn-detection-test", vad=vad, stt=stt))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=turn-detection-test")
        created = json.loads((await ws.receive()).data)
        assert created["session"]["turn_detection"] == {"type": "server_vad"}
    finally:
        pipeline_registry.unregister("turn-detection-test")


async def test_session_created_turn_detection_null_without_vad(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="no-vad-turn-detection-test", llm=_StubLLM()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=no-vad-turn-detection-test")
        created = json.loads((await ws.receive()).data)
        assert created["session"]["turn_detection"] is None
    finally:
        pipeline_registry.unregister("no-vad-turn-detection-test")


async def test_speech_to_text_shape_emits_no_audio_events(aiohttp_client):
    vad = _StubVAD(start_at_frame=1, end_at_frame=2)
    stt = _StubSTT(fixed_text="speech to text only")
    pipeline_registry.register(PipelineConfig(name="speech-to-text-shape-test", llm=_StubLLM(), vad=vad, stt=stt))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=speech-to-text-shape-test")
        created = json.loads((await ws.receive()).data)
        assert created["session"]["modalities"]["output"] == ["text"]

        await ws.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(_silence_frame()).decode("ascii")}
        )

        events_received = []
        while True:
            msg = json.loads((await ws.receive()).data)
            events_received.append(msg["type"])
            if msg["type"] == "response.done":
                break

        assert "conversation.item.input_audio_transcription.completed" in events_received
        assert "response.output_text.delta" in events_received
        assert not any(t.startswith("response.output_audio") for t in events_received)
    finally:
        pipeline_registry.unregister("speech-to-text-shape-test")


async def test_text_to_text_shape_reports_text_only_modalities(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="text-to-text-shape-test", llm=_StubLLM()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=text-to-text-shape-test")
        created = json.loads((await ws.receive()).data)
        assert created["session"]["modalities"]["input"] == ["text"]
        assert created["session"]["modalities"]["output"] == ["text"]
        assert created["session"]["turn_detection"] is None

        await ws.send_json(
            {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}}
        )
        await ws.receive()  # item.added
        await ws.receive()  # item.done
        await ws.send_json({"type": "response.create"})

        events_received = []
        while True:
            msg = json.loads((await ws.receive()).data)
            events_received.append(msg["type"])
            if msg["type"] == "response.done":
                break

        assert "response.output_text.delta" in events_received
        assert not any("audio" in t for t in events_received)
    finally:
        pipeline_registry.unregister("text-to-text-shape-test")


class _TwoUtteranceStubVAD:
    """Fires two separate speech_started/speech_ended cycles, to prove
    continuous-transcription pipelines handle a second utterance correctly
    after the first, with no response cycle in between to reset anything."""

    BOUNDARIES = {2: "started", 4: "ended", 7: "started", 9: "ended"}

    def __init__(self):
        self._count = 0

    async def analyze(self, audio_chunk):
        self._count += 1
        boundary = self.BOUNDARIES.get(self._count)
        return VADResult(
            speech_probability=0.9,
            speech_started=(boundary == "started"),
            speech_ended=(boundary == "ended"),
        )


class _CapturingStubLLM:
    """Captures the messages list from its most recent generate() call, so
    a test can inspect exactly what each pipeline's instructions produced,
    even though both pipelines share this one provider instance."""

    def __init__(self):
        self.last_messages = None

    async def generate(self, messages, options):
        self.last_messages = messages
        yield GenerationDelta(text="ok", finished=False)
        yield GenerationDelta(text="", finished=True)


async def test_shared_llm_applies_each_pipelines_own_instructions(aiohttp_client):
    shared_llm = _CapturingStubLLM()
    pipeline_registry.register(PipelineConfig(name="shared-a-test", llm=shared_llm, instructions="Be a pirate."))
    pipeline_registry.register(PipelineConfig(name="shared-b-test", llm=shared_llm, instructions="Be a robot."))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)

        ws_a = await client.ws_connect("/v1/realtime?model=shared-a-test")
        await ws_a.receive()  # session.created
        await ws_a.send_json(
            {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}}
        )
        await ws_a.receive()  # item.added
        await ws_a.receive()  # item.done
        await ws_a.send_json({"type": "response.create"})
        await ws_a.receive()  # response.created
        await ws_a.receive()  # output_text.delta -- by now generate() ran and captured messages
        assert shared_llm.last_messages[0].role == "system"
        assert shared_llm.last_messages[0].content == "Be a pirate."

        ws_b = await client.ws_connect("/v1/realtime?model=shared-b-test")
        await ws_b.receive()
        await ws_b.send_json(
            {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}}
        )
        await ws_b.receive()
        await ws_b.receive()
        await ws_b.send_json({"type": "response.create"})
        await ws_b.receive()
        await ws_b.receive()
        assert shared_llm.last_messages[0].role == "system"
        assert shared_llm.last_messages[0].content == "Be a robot."
    finally:
        pipeline_registry.unregister("shared-a-test")
        pipeline_registry.unregister("shared-b-test")


async def test_session_appears_in_registry_while_connected_and_not_after(aiohttp_client):
    from comfyui_realtime.session_registry import session_registry

    pipeline_registry.register(PipelineConfig(name="session-lifecycle-test", llm=_StubLLM()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        before_count = session_registry.count()

        ws = await client.ws_connect("/v1/realtime?model=session-lifecycle-test")
        await ws.receive()  # session.created
        assert session_registry.count() == before_count + 1
        records = [r for r in session_registry.list() if r.pipeline_name == "session-lifecycle-test"]
        assert len(records) == 1

        await ws.close()
        # Give the server's message loop a moment to observe the close and
        # run its cleanup.
        import asyncio

        await asyncio.sleep(0.1)
        assert session_registry.count() == before_count
    finally:
        pipeline_registry.unregister("session-lifecycle-test")


async def test_continuous_transcription_shape_streams_multiple_utterances(aiohttp_client):
    vad = _TwoUtteranceStubVAD()
    stt = _StubSTT(fixed_text="continuous transcript text")
    pipeline_registry.register(PipelineConfig(name="continuous-transcription-shape-test", vad=vad, stt=stt))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=continuous-transcription-shape-test")
        created = json.loads((await ws.receive()).data)
        assert created["session"]["modalities"]["input"] == ["text", "audio"]
        assert created["session"]["modalities"]["output"] == ["text"]
        assert created["session"]["turn_detection"] == {"type": "server_vad"}

        # 5 appends x 2 frames/append (per Phase 2's established frame
        # arithmetic for this silence chunk size) covers boundary frames 2,4,7,9.
        for _ in range(5):
            await ws.send_json(
                {"type": "input_audio_buffer.append", "audio": base64.b64encode(_silence_frame()).decode("ascii")}
            )

        events_received = []
        completed_count = 0
        for _ in range(20):
            msg = json.loads((await ws.receive()).data)
            events_received.append(msg["type"])
            if msg["type"] == "conversation.item.input_audio_transcription.completed":
                completed_count += 1
                if completed_count == 2:
                    break

        assert completed_count == 2, f"expected 2 transcription cycles, saw events: {events_received}"
        assert events_received.count("input_audio_buffer.speech_started") == 2
        assert events_received.count("input_audio_buffer.speech_stopped") == 2
        assert not any(t.startswith("response.") for t in events_received)
    finally:
        pipeline_registry.unregister("continuous-transcription-shape-test")


class _MultiVoiceStubTTS:
    """A stub TTS provider exposing two voices, recording which voice
    synthesize() was actually called with -- lets tests observe the
    effect of session.voice_override without needing real Piper models."""

    output_sample_rate = 24000
    output_format = "pcm16"

    def __init__(self):
        self.last_voice_used = None

    async def synthesize(self, text_stream, voice=None):
        self.last_voice_used = voice
        async for text in text_stream:
            yield f"audio-for:{text}".encode("ascii")

    def list_voices(self):
        from comfyui_realtime.providers.base import VoiceInfo

        return [VoiceInfo(id="voice-a", name="voice-a"), VoiceInfo(id="voice-b", name="voice-b")]


async def test_session_created_includes_the_pipelines_default_voice(aiohttp_client):
    tts = _MultiVoiceStubTTS()
    pipeline_registry.register(PipelineConfig(name="voice-test-created", llm=_StubLLM(), tts=tts, voice="voice-a"))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=voice-test-created")
        created = json.loads((await ws.receive()).data)
        assert created["session"]["voice"] == "voice-a"
    finally:
        pipeline_registry.unregister("voice-test-created")


async def test_session_update_with_a_valid_voice_overrides_synthesis(aiohttp_client):
    tts = _MultiVoiceStubTTS()
    pipeline_registry.register(PipelineConfig(name="voice-test-override", llm=_StubLLM(), tts=tts, voice="voice-a"))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=voice-test-override")
        await ws.receive()  # session.created

        await ws.send_json({"type": "session.update", "session": {"voice": "voice-b"}})
        updated = json.loads((await ws.receive()).data)
        assert updated["type"] == "session.updated"
        assert updated["session"]["voice"] == "voice-b"

        await ws.send_json(
            {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}}
        )
        await ws.receive()  # item.added
        await ws.receive()  # item.done
        await ws.send_json({"type": "response.create"})

        # Drain events until response.done -- by then synthesize() has run.
        while True:
            msg = json.loads((await ws.receive()).data)
            if msg["type"] == "response.done":
                break
        assert tts.last_voice_used == "voice-b"
    finally:
        pipeline_registry.unregister("voice-test-override")


async def test_session_update_with_an_unknown_voice_returns_an_error_and_keeps_the_default(aiohttp_client):
    tts = _MultiVoiceStubTTS()
    pipeline_registry.register(PipelineConfig(name="voice-test-unknown", llm=_StubLLM(), tts=tts, voice="voice-a"))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=voice-test-unknown")
        await ws.receive()  # session.created

        await ws.send_json({"type": "session.update", "session": {"voice": "voice-does-not-exist"}})
        error = json.loads((await ws.receive()).data)
        assert error["type"] == "error"
        assert error["error"]["code"] == "unknown_voice"

        await ws.send_json(
            {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}}
        )
        await ws.receive()
        await ws.receive()
        await ws.send_json({"type": "response.create"})
        while True:
            msg = json.loads((await ws.receive()).data)
            if msg["type"] == "response.done":
                break
        assert tts.last_voice_used == "voice-a"  # the pipeline's original default, not the rejected request
    finally:
        pipeline_registry.unregister("voice-test-unknown")


async def test_two_concurrent_sessions_on_one_shared_pipeline_never_cross_talk_on_voice(aiohttp_client):
    """The single most important property this body of work exists to
    guarantee: SessionState.voice_override is per-connection, not shared
    state on PipelineConfig. Two sessions connect to the SAME pipeline
    object, set DIFFERENT voices, and interleave their response.create
    calls -- if voice_override were ever stored on the shared
    PipelineConfig instead of SessionState, this test would catch it as
    one session's voice leaking into the other's synthesize() call."""
    tts = _MultiVoiceStubTTS()
    pipeline_registry.register(PipelineConfig(name="voice-test-cross-talk", llm=_StubLLM(), tts=tts, voice="voice-a"))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)

        ws_a = await client.ws_connect("/v1/realtime?model=voice-test-cross-talk")
        await ws_a.receive()  # session.created
        ws_b = await client.ws_connect("/v1/realtime?model=voice-test-cross-talk")
        await ws_b.receive()  # session.created

        # Both sessions are now connected to the literal same PipelineConfig
        # object (one registry entry, one pipeline_name) -- confirmed by
        # registering only once above.
        await ws_a.send_json({"type": "session.update", "session": {"voice": "voice-a"}})
        await ws_a.receive()  # session.updated
        await ws_b.send_json({"type": "session.update", "session": {"voice": "voice-b"}})
        await ws_b.receive()  # session.updated

        # Interleave: trigger session B's response BEFORE session A's, so
        # any shared/global state would show up as session A's later
        # response incorrectly using voice-b.
        async def run_conversation(ws, expected_text):
            await ws.send_json(
                {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": expected_text}]}}
            )
            await ws.receive()  # item.added
            await ws.receive()  # item.done
            await ws.send_json({"type": "response.create"})
            while True:
                msg = json.loads((await ws.receive()).data)
                if msg["type"] == "response.done":
                    break

        await run_conversation(ws_b, "from b")
        assert tts.last_voice_used == "voice-b"

        await run_conversation(ws_a, "from a")
        assert tts.last_voice_used == "voice-a"  # NOT voice-b -- this is the property under test

        # And confirm it still holds the other direction, in case ordering
        # alone happened to mask a bug:
        await run_conversation(ws_b, "from b again")
        assert tts.last_voice_used == "voice-b"
    finally:
        pipeline_registry.unregister("voice-test-cross-talk")


async def test_session_update_with_an_unknown_voice_does_not_clobber_an_existing_valid_override(aiohttp_client):
    """Companion to test_session_update_with_an_unknown_voice_returns_an_error_and_keeps_the_default
    (Task 2), which only proves the no-prior-override case: reject an
    invalid voice, the pipeline's *default* survives. That left a gap,
    found during Task 2's review: if a session already has a valid
    override in place and then sends an unrelated invalid voice request,
    does the *original override* survive, or does rejection silently fall
    back to the pipeline default (or worse, get clobbered)? Verified
    correct via a one-off scratch test during Task 2's review; this makes
    it a permanent regression test."""
    tts = _MultiVoiceStubTTS()
    pipeline_registry.register(PipelineConfig(name="voice-test-reject-keeps-override", llm=_StubLLM(), tts=tts, voice="voice-a"))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=voice-test-reject-keeps-override")
        await ws.receive()  # session.created

        # Establish a valid override different from the pipeline's default.
        await ws.send_json({"type": "session.update", "session": {"voice": "voice-b"}})
        updated = json.loads((await ws.receive()).data)
        assert updated["type"] == "session.updated"
        assert updated["session"]["voice"] == "voice-b"

        # Now send an unrelated invalid voice request.
        await ws.send_json({"type": "session.update", "session": {"voice": "voice-does-not-exist"}})
        error = json.loads((await ws.receive()).data)
        assert error["type"] == "error"
        assert error["error"]["code"] == "unknown_voice"

        await ws.send_json(
            {"type": "conversation.item.create", "item": {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}}
        )
        await ws.receive()  # item.added
        await ws.receive()  # item.done
        await ws.send_json({"type": "response.create"})
        while True:
            msg = json.loads((await ws.receive()).data)
            if msg["type"] == "response.done":
                break
        # The ORIGINAL override survives -- not the pipeline's default
        # ("voice-a"), and not silently reset by the rejected request.
        assert tts.last_voice_used == "voice-b"
    finally:
        pipeline_registry.unregister("voice-test-reject-keeps-override")


async def test_session_update_voice_is_a_silent_noop_when_pipeline_has_no_tts(aiohttp_client):
    pipeline_registry.register(PipelineConfig(name="voice-test-no-tts", llm=_StubLLM()))
    try:
        app = await _build_app()
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/v1/realtime?model=voice-test-no-tts")
        await ws.receive()  # session.created

        await ws.send_json({"type": "session.update", "session": {"voice": "anything"}})
        updated = json.loads((await ws.receive()).data)
        assert updated["type"] == "session.updated"  # acked, not errored
        assert updated["session"]["voice"] is None
    finally:
        pipeline_registry.unregister("voice-test-no-tts")
