"""Realtime WebSocket session handler.

Implements: session creation with pipeline-not-found handling, and
conversation.item.create, which appends to a per-connection conversation
history and acknowledges with conversation.item.added/done. Adding an item
to history never by itself triggers generation -- response.create is a
separate client message that reads this history.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging

from aiohttp import WSMsgType, web

from .. import registry
from ..engine import dialects, events
from ..engine.audio_framer import AudioFrameBuffer
from ..engine.resample import resample_pcm16
from ..engine.sentence_chunker import SentenceChunker
from ..engine.session_state import CancelledResponseRecord, SessionState
from ..providers.base import ChatMessage, GenerationOptions
from ..session_registry import session_registry

logger = logging.getLogger("comfyui_realtime")


async def realtime_websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """Get the pipeline name from the query string. If not provided, default to "echo"."""
    pipeline_name = request.rel_url.query.get("model", "echo")
    dialect = dialects.select_dialect(request.headers.get("OpenAI-Beta"))

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    session_id = events.new_id("sess")
    session = SessionState(session_id=session_id)
    # Register before the pipeline lookup, not after -- closes a TOCTOU race:
    # the conservative unload rule's session_registry.count() check could
    # otherwise see zero sessions and unload a provider this connection is
    # about to start using, in the window between looking the pipeline up
    # and registering. Only the pipeline_name string is needed here; this
    # connection hasn't captured any provider reference yet, so registering
    # early is always safe.
    session_registry.register(session_id, pipeline_name, session)
    try:
        pipeline = registry.pipeline_registry.get(pipeline_name)
        if pipeline is None:
            # Accept the handshake, send a protocol error, then close with the
            # application close code. A plain HTTP 404 surfaces poorly in
            # Realtime SDKs, which expect post-handshake error events.
            await ws.send_json(
                dialect.serialize(
                    events.ErrorEvent(
                        code="pipeline_not_found",
                        message=f"No pipeline registered as '{pipeline_name}'",
                    )
                )
            )
            await ws.close(code=4404, message=b"pipeline_not_found")
            return ws

        await ws.send_json(
            dialect.serialize(
                events.SessionCreatedEvent(
                    session_id=session_id,
                    pipeline_name=pipeline_name,
                    modalities_input=pipeline.modalities_input,
                    modalities_output=pipeline.modalities_output,
                    turn_detection={"type": "server_vad"} if pipeline.vad is not None else None,
                    voice=pipeline.voice,
                )
            )
        )

        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await _handle_client_event(ws, dialect, pipeline, session, json.loads(msg.data))
            elif msg.type == WSMsgType.ERROR:
                logger.warning("realtime websocket closed with exception %s", ws.exception())
    finally:
        session_registry.unregister(session_id)

    return ws


async def _handle_client_event(
    ws: web.WebSocketResponse,
    dialect: dialects.DialectSerializer,
    pipeline: registry.PipelineConfig,
    session: SessionState,
    data: dict,
) -> None:
    event_type = data.get("type")
    if event_type == "conversation.item.create":
        await _handle_item_create(ws, dialect, session.conversation_history, data)
    elif event_type == "response.create":
        if pipeline.llm is None:
            await ws.send_json(
                dialect.serialize(
                    events.ErrorEvent(
                        code="llm_not_configured",
                        message="This pipeline has no LLM provider configured; response.create is not supported.",
                    )
                )
            )
            return
        _start_response_task(ws, dialect, pipeline, session)
    elif event_type == "input_audio_buffer.append":
        await _handle_audio_append(ws, dialect, pipeline, session, data)
    elif event_type == "input_audio_buffer.commit":
        await _handle_audio_commit(ws, dialect, pipeline, session)
    elif event_type == "input_audio_buffer.clear":
        await _handle_audio_clear(ws, dialect, session)
    elif event_type == "response.cancel":
        await _cancel_active_response(session)
    elif event_type == "conversation.item.truncate":
        await _handle_conversation_item_truncate(ws, dialect, session, data)
    elif event_type == "session.update":
        requested_voice = data.get("session", {}).get("voice")
        if requested_voice is not None and pipeline.tts is not None:
            known_voice_ids = {v.id for v in pipeline.tts.list_voices()}
            if requested_voice not in known_voice_ids:
                await ws.send_json(
                    dialect.serialize(
                        events.ErrorEvent(
                            code="unknown_voice",
                            message=f"'{requested_voice}' is not a voice this pipeline's TTS provider has loaded.",
                        )
                    )
                )
                return
            session.voice_override = requested_voice
        # Every other field is read and applied nowhere -- live reconfiguration
        # of an active session is an explicit non-goal. voice is the one
        # deliberate, narrow exception, validated
        # above against the pipeline's actual loaded voices before being
        # accepted -- never blindly trusted, since synthesize() does an
        # unguarded dict lookup on it.
        await ws.send_json(
            dialect.serialize(
                events.SessionUpdatedEvent(
                    session_id=session.session_id,
                    pipeline_name=pipeline.name,
                    modalities_input=pipeline.modalities_input,
                    modalities_output=pipeline.modalities_output,
                    turn_detection={"type": "server_vad"} if pipeline.vad is not None else None,
                    voice=session.voice_override or pipeline.voice,
                )
            )
        )
    # Other event types are silently ignored, matching the existing
    # behavior for events outside the implemented subset.


async def _cancel_active_response(session: SessionState) -> None:
    task = session.active_response_task
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _start_response_task(
    ws: web.WebSocketResponse,
    dialect: dialects.DialectSerializer,
    pipeline: registry.PipelineConfig,
    session: SessionState,
) -> None:
    if session.active_response_task is not None and not session.active_response_task.done():
        logger.warning("response.create received while a response is already active; ignoring")
        return

    task = asyncio.create_task(_handle_response_create(ws, dialect, pipeline, session))

    def _on_done(t: asyncio.Task) -> None:
        if session.active_response_task is t:
            session.active_response_task = None
        if not t.cancelled() and t.exception() is not None:
            logger.error("response.create task raised an unhandled exception: %r", t.exception())

    task.add_done_callback(_on_done)
    session.active_response_task = task


async def _handle_item_create(
    ws: web.WebSocketResponse,
    dialect: dialects.DialectSerializer,
    conversation_history: list[ChatMessage],
    data: dict,
) -> None:
    item = data.get("item", {})
    role = item.get("role", "user")
    text = _extract_text(data)
    item_id = events.new_id("item")

    conversation_history.append(ChatMessage(role=role, content=text))

    await ws.send_json(
        dialect.serialize(events.ConversationItemAddedEvent(item_id=item_id, role=role, text=text))
    )
    await ws.send_json(
        dialect.serialize(events.ConversationItemDoneEvent(item_id=item_id, role=role, text=text))
    )


def _extract_text(data: dict) -> str:
    item = data.get("item", {})
    for part in item.get("content", []):
        if part.get("type") in ("input_text", "text"):
            return part.get("text", "")
    return ""


async def _handle_audio_append(
    ws: web.WebSocketResponse,
    dialect: dialects.DialectSerializer,
    pipeline: registry.PipelineConfig,
    session: SessionState,
    data: dict,
) -> None:
    if pipeline.stt is None:
        await ws.send_json(
            dialect.serialize(
                events.ErrorEvent(
                    code="audio_input_not_supported",
                    message="This pipeline has no STT provider configured.",
                )
            )
        )
        return

    wire_audio = session.pending_wire_audio + base64.b64decode(data.get("audio", ""))
    # pcm16 is 2 bytes/sample; input_audio_buffer.append chunk sizes are
    # arbitrary (not guaranteed sample-aligned), so carry over a trailing
    # odd byte to prepend to the next chunk rather than handing
    # resample_pcm16 a buffer that isn't a whole number of samples.
    if len(wire_audio) % 2:
        session.pending_wire_audio = wire_audio[-1:]
        wire_audio = wire_audio[:-1]
    else:
        session.pending_wire_audio = b""
    if not wire_audio:
        return
    audio_16k = resample_pcm16(wire_audio, from_rate=24000, to_rate=16000)
    session.utterance_audio_16k += audio_16k

    if pipeline.vad is None:
        return  # manual commit-based turn detection; nothing more per chunk

    if session.audio_framer is None:
        session.audio_framer = AudioFrameBuffer()

    for frame in session.audio_framer.push(audio_16k):
        result = await pipeline.vad.analyze(frame)
        if result.speech_started and not session.in_speech:
            session.in_speech = True
            session.pending_item_id = events.new_id("item")
            await _cancel_active_response(session)
            await ws.send_json(
                dialect.serialize(events.InputAudioBufferSpeechStartedEvent(item_id=session.pending_item_id))
            )
        elif result.speech_ended and session.in_speech:
            session.in_speech = False
            await ws.send_json(
                dialect.serialize(events.InputAudioBufferSpeechStoppedEvent(item_id=session.pending_item_id))
            )
            await _finalize_utterance(ws, dialect, pipeline, session)


async def _handle_audio_commit(
    ws: web.WebSocketResponse,
    dialect: dialects.DialectSerializer,
    pipeline: registry.PipelineConfig,
    session: SessionState,
) -> None:
    if pipeline.stt is None or pipeline.vad is not None:
        # commit is only meaningful for the no-VAD manual-turn-boundary case
        return
    session.pending_item_id = events.new_id("item")
    await _finalize_utterance(ws, dialect, pipeline, session)


async def _handle_audio_clear(
    ws: web.WebSocketResponse, dialect: dialects.DialectSerializer, session: SessionState
) -> None:
    session.utterance_audio_16k = b""
    session.audio_framer = None
    session.pending_wire_audio = b""
    session.in_speech = False
    session.pending_item_id = None
    await ws.send_json(dialect.serialize(events.InputAudioBufferClearedEvent()))


async def _finalize_utterance(
    ws: web.WebSocketResponse,
    dialect: dialects.DialectSerializer,
    pipeline: registry.PipelineConfig,
    session: SessionState,
) -> None:
    item_id = session.pending_item_id
    audio_for_stt = session.utterance_audio_16k
    session.utterance_audio_16k = b""
    session.pending_item_id = None

    await ws.send_json(dialect.serialize(events.InputAudioBufferCommittedEvent(item_id=item_id)))

    try:
        result = await pipeline.stt.transcribe(audio_for_stt)
    except Exception as exc:  # noqa: BLE001 -- reported to the client as a protocol event, not swallowed silently
        await ws.send_json(
            dialect.serialize(
                events.ConversationItemInputAudioTranscriptionFailedEvent(item_id=item_id, error_message=str(exc))
            )
        )
        return

    await ws.send_json(
        dialect.serialize(
            events.ConversationItemInputAudioTranscriptionCompletedEvent(item_id=item_id, transcript=result.text)
        )
    )

    if result.text.strip():
        session.conversation_history.append(ChatMessage(role="user", content=result.text))
        if pipeline.llm is not None:
            _start_response_task(ws, dialect, pipeline, session)
        # else: continuous-transcription shape -- the transcript is the whole
        # point; no LLM means no response is ever expected, and that's
        # correct, not an error.


async def _drain_queue_as_stream(queue: "asyncio.Queue[str | None]", on_item):
    while True:
        item = await queue.get()
        if item is None:
            return
        await on_item(item)
        yield item


async def _handle_response_create(
    ws: web.WebSocketResponse,
    dialect: dialects.DialectSerializer,
    pipeline: registry.PipelineConfig,
    session: SessionState,
) -> None:
    conversation_history = session.conversation_history
    response_id = events.new_id("resp")
    item_id = events.new_id("item")

    await ws.send_json(dialect.serialize(events.ResponseCreatedEvent(response_id=response_id)))

    messages = list(conversation_history)
    if pipeline.instructions:
        messages = [ChatMessage(role="system", content=pipeline.instructions)] + messages

    sentence_queue: "asyncio.Queue[str | None]" = asyncio.Queue()
    chunker = SentenceChunker()
    text_parts: list[str] = []
    delivered_audio_bytes = 0

    async def run_llm() -> None:
        options = GenerationOptions(temperature=pipeline.temperature)
        async for delta in pipeline.llm.generate(messages, options):
            if delta.finished:
                break
            text_parts.append(delta.text)
            await ws.send_json(
                dialect.serialize(
                    events.ResponseOutputTextDeltaEvent(
                        response_id=response_id, item_id=item_id, output_index=0, content_index=0, delta=delta.text,
                    )
                )
            )
            for sentence in chunker.feed(delta.text):
                await sentence_queue.put(sentence)
        trailing = chunker.flush()
        if trailing:
            await sentence_queue.put(trailing)
        await sentence_queue.put(None)
        await ws.send_json(
            dialect.serialize(
                events.ResponseOutputTextDoneEvent(
                    response_id=response_id, item_id=item_id, output_index=0, content_index=0,
                    text="".join(text_parts),
                )
            )
        )

    async def run_tts() -> None:
        nonlocal delivered_audio_bytes
        if pipeline.tts is None:
            return
        transcript_parts: list[str] = []

        async def on_sentence(sentence: str) -> None:
            transcript_parts.append(sentence)
            await ws.send_json(
                dialect.serialize(
                    events.ResponseOutputAudioTranscriptDeltaEvent(
                        response_id=response_id, item_id=item_id, output_index=0, content_index=0, delta=sentence,
                    )
                )
            )

        text_stream = _drain_queue_as_stream(sentence_queue, on_sentence)
        async for audio_chunk in pipeline.tts.synthesize(text_stream, voice=session.voice_override or pipeline.voice):
            delivered_audio_bytes += len(audio_chunk)
            await ws.send_json(
                dialect.serialize(
                    events.ResponseOutputAudioDeltaEvent(
                        response_id=response_id, item_id=item_id, output_index=0, content_index=0,
                        delta_bytes=audio_chunk,
                    )
                )
            )
        await ws.send_json(
            dialect.serialize(
                events.ResponseOutputAudioDoneEvent(
                    response_id=response_id, item_id=item_id, output_index=0, content_index=0
                )
            )
        )
        await ws.send_json(
            dialect.serialize(
                events.ResponseOutputAudioTranscriptDoneEvent(
                    response_id=response_id, item_id=item_id, output_index=0, content_index=0,
                    transcript="".join(transcript_parts),
                )
            )
        )

    # run_llm/run_tts must be explicit asyncio.Task objects, not bare
    # coroutines passed straight to gather(). asyncio.gather()'s own
    # cancellation future can resolve to cancelled -- returning control to
    # whatever awaits it -- BEFORE its children's own cancellation cleanup
    # (each provider's finally: await bridge.aclose(); lock.release()) has
    # actually finished running. The except block below explicitly re-awaits
    # any not-yet-done child task before this function itself finishes
    # unwinding, which guarantees the provider lock is free by the time
    # _cancel_active_response's `await task` returns to its caller.
    llm_task = asyncio.create_task(run_llm())
    tts_task = asyncio.create_task(run_tts()) if pipeline.tts is not None else None

    try:
        if tts_task is not None:
            await asyncio.gather(llm_task, tts_task)
        else:
            await llm_task
            await sentence_queue.get()

        conversation_history.append(ChatMessage(role="assistant", content="".join(text_parts)))
        await ws.send_json(dialect.serialize(events.ResponseDoneEvent(response_id=response_id)))
    except asyncio.CancelledError:
        pending = [t for t in (llm_task, tts_task) if t is not None and not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        session.last_cancelled_response = CancelledResponseRecord(
            item_id=item_id,
            delivered_text="".join(text_parts),
            delivered_audio_bytes=delivered_audio_bytes,
            output_sample_rate=pipeline.tts.output_sample_rate if pipeline.tts is not None else 24000,
        )
        conversation_history.append(ChatMessage(role="assistant", content="".join(text_parts)))
        await ws.send_json(
            dialect.serialize(events.ResponseDoneEvent(response_id=response_id, status="cancelled"))
        )
        raise


async def _handle_conversation_item_truncate(
    ws: web.WebSocketResponse, dialect: dialects.DialectSerializer, session: SessionState, data: dict
) -> None:
    item_id = data.get("item_id")
    audio_end_ms = data.get("audio_end_ms", 0)
    record = session.last_cancelled_response

    if record is not None and record.item_id == item_id and record.delivered_audio_bytes > 0:
        delivered_ms = (record.delivered_audio_bytes / 2 / record.output_sample_rate) * 1000
        if delivered_ms > 0 and audio_end_ms < delivered_ms:
            fraction = max(0.0, min(1.0, audio_end_ms / delivered_ms))
            truncated_text = record.delivered_text[: int(len(record.delivered_text) * fraction)]
            for msg in reversed(session.conversation_history):
                if msg.role == "assistant":
                    msg.content = truncated_text
                    break
        session.last_cancelled_response = None

    await ws.send_json(dialect.serialize(events.ConversationItemTruncatedEvent(item_id=item_id)))
