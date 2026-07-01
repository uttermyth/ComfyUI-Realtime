"""Internal canonical event model.

The engine never builds raw protocol dicts directly -- it builds one of
these dataclasses and hands it to a dialect serializer (dialects.py), which
is the only place wire-format names live. This is what lets GA and beta
dialects (and future protocol drift) stay confined to one module.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass


def new_id(prefix: str) -> str:
    """IDs in OpenAI's Realtime style: '<prefix>_<24 hex chars>'."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


@dataclass
class SessionCreatedEvent:
    session_id: str
    pipeline_name: str
    modalities_input: list[str]
    modalities_output: list[str]
    turn_detection: dict | None = None
    voice: str | None = None


@dataclass
class SessionUpdatedEvent:
    """Acknowledgment-only response to a client's session.update -- live
    reconfiguration of an active session is unsupported. Mirrors
    SessionCreatedEvent's fields exactly, echoing the pipeline's real,
    fixed, registration-time config -- never anything read from the
    client's session.update payload."""

    session_id: str
    pipeline_name: str
    modalities_input: list[str]
    modalities_output: list[str]
    turn_detection: dict | None = None
    voice: str | None = None


@dataclass
class ResponseCreatedEvent:
    response_id: str


@dataclass
class ResponseOutputTextDeltaEvent:
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


@dataclass
class ResponseOutputTextDoneEvent:
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    text: str


@dataclass
class ResponseDoneEvent:
    response_id: str
    status: str = "completed"


@dataclass
class ErrorEvent:
    code: str
    message: str
    event_id: str | None = None


@dataclass
class ConversationItemAddedEvent:
    item_id: str
    role: str
    text: str


@dataclass
class ConversationItemDoneEvent:
    item_id: str
    role: str
    text: str


@dataclass
class ResponseOutputAudioDeltaEvent:
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta_bytes: bytes


@dataclass
class ResponseOutputAudioDoneEvent:
    response_id: str
    item_id: str
    output_index: int
    content_index: int


@dataclass
class ResponseOutputAudioTranscriptDeltaEvent:
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


@dataclass
class ResponseOutputAudioTranscriptDoneEvent:
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    transcript: str


@dataclass
class InputAudioBufferSpeechStartedEvent:
    item_id: str


@dataclass
class InputAudioBufferSpeechStoppedEvent:
    item_id: str


@dataclass
class InputAudioBufferCommittedEvent:
    item_id: str


@dataclass
class InputAudioBufferClearedEvent:
    pass


@dataclass
class ConversationItemInputAudioTranscriptionCompletedEvent:
    item_id: str
    transcript: str


@dataclass
class ConversationItemInputAudioTranscriptionFailedEvent:
    item_id: str
    error_message: str


@dataclass
class ConversationItemTruncatedEvent:
    item_id: str
