"""Per-connection mutable state for one realtime session.

Replaces threading conversation_history (and, from this point on, the
active response task and later the audio-input buffering state) as
separate loose parameters through every websocket_handler helper function
-- this object grows with the protocol surface instead of every function
signature growing each time a new piece of per-connection state is needed.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ..providers.base import ChatMessage
from .audio_framer import AudioFrameBuffer


@dataclass
class CancelledResponseRecord:
    item_id: str
    delivered_text: str
    delivered_audio_bytes: int
    output_sample_rate: int


@dataclass
class SessionState:
    conversation_history: list[ChatMessage] = field(default_factory=list)
    active_response_task: asyncio.Task | None = None
    audio_framer: AudioFrameBuffer | None = None
    utterance_audio_16k: bytes = b""
    # input_audio_buffer.append chunks arrive at whatever byte length the
    # client happens to send -- not necessarily a whole number of pcm16
    # samples. resample_pcm16 requires a whole number of samples, so any
    # odd trailing byte from one append call is carried over and prepended
    # to the next, rather than fed to the resampler directly (which would
    # raise on an odd-length buffer).
    pending_wire_audio: bytes = b""
    in_speech: bool = False
    pending_item_id: str | None = None
    last_cancelled_response: CancelledResponseRecord | None = None
    # Set at construction from the same session_id sent in session.created,
    # so the session.update ack handler can echo the session's real id
    # without threading it through every handler's parameters.
    session_id: str | None = None
    # Set by session.update's one deliberate exception to its otherwise
    # ack-only design -- never read or written by anything except the
    # session.update handler and the TTS call site in websocket_handler.py.
    # Per-session, not on PipelineConfig: a shared pipeline may serve
    # multiple concurrent sessions, and mutating the pipeline's own voice
    # would leak one session's choice into every other session connected to it.
    voice_override: str | None = None
