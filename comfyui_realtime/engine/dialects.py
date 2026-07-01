"""Wire-format serializers.

OpenAI's Realtime API went GA in August 2025 and renamed several response
events (e.g. response.audio.delta -> response.output_audio.delta). The old
names are the "beta" dialect, selected by clients sending the header
`OpenAI-Beta: realtime=v1`. This module is the *only* place those wire-format
strings appear -- the engine itself only ever builds events.py dataclasses.
"""
from __future__ import annotations

import base64
from typing import Protocol

from . import events


class DialectSerializer(Protocol):
    name: str

    def serialize(self, event: object) -> dict:
        ...


class GADialectSerializer:
    name = "ga"

    def serialize(self, event: object) -> dict:
        if isinstance(event, events.SessionCreatedEvent):
            return {
                "type": "session.created",
                "session": {
                    "id": event.session_id,
                    "model": event.pipeline_name,
                    "modalities": {
                        "input": event.modalities_input,
                        "output": event.modalities_output,
                    },
                    "turn_detection": event.turn_detection,
                    "voice": event.voice,
                },
            }
        if isinstance(event, events.ResponseCreatedEvent):
            return {
                "type": "response.created",
                "response": {"id": event.response_id, "status": "in_progress"},
            }
        if isinstance(event, events.ResponseOutputTextDeltaEvent):
            return {
                "type": "response.output_text.delta",
                "response_id": event.response_id,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
                "delta": event.delta,
            }
        if isinstance(event, events.ResponseOutputTextDoneEvent):
            return {
                "type": "response.output_text.done",
                "response_id": event.response_id,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
                "text": event.text,
            }
        if isinstance(event, events.ResponseDoneEvent):
            return {
                "type": "response.done",
                "response": {"id": event.response_id, "status": event.status},
            }
        if isinstance(event, events.ErrorEvent):
            return {
                "type": "error",
                "error": {"code": event.code, "message": event.message},
                "event_id": event.event_id,
            }
        if isinstance(event, events.ConversationItemAddedEvent):
            return {
                "type": "conversation.item.added",
                "item": {
                    "id": event.item_id,
                    "role": event.role,
                    "content": [{"type": "input_text", "text": event.text}],
                },
            }
        if isinstance(event, events.ConversationItemDoneEvent):
            return {
                "type": "conversation.item.done",
                "item": {
                    "id": event.item_id,
                    "role": event.role,
                    "content": [{"type": "input_text", "text": event.text}],
                },
            }
        if isinstance(event, events.ResponseOutputAudioDeltaEvent):
            return {
                "type": "response.output_audio.delta",
                "response_id": event.response_id,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
                "delta": base64.b64encode(event.delta_bytes).decode("ascii"),
            }
        if isinstance(event, events.ResponseOutputAudioDoneEvent):
            return {
                "type": "response.output_audio.done",
                "response_id": event.response_id,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
            }
        if isinstance(event, events.ResponseOutputAudioTranscriptDeltaEvent):
            return {
                "type": "response.output_audio_transcript.delta",
                "response_id": event.response_id,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
                "delta": event.delta,
            }
        if isinstance(event, events.ResponseOutputAudioTranscriptDoneEvent):
            return {
                "type": "response.output_audio_transcript.done",
                "response_id": event.response_id,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
                "transcript": event.transcript,
            }
        if isinstance(event, events.InputAudioBufferSpeechStartedEvent):
            return {"type": "input_audio_buffer.speech_started", "item_id": event.item_id}
        if isinstance(event, events.InputAudioBufferSpeechStoppedEvent):
            return {"type": "input_audio_buffer.speech_stopped", "item_id": event.item_id}
        if isinstance(event, events.InputAudioBufferCommittedEvent):
            return {"type": "input_audio_buffer.committed", "item_id": event.item_id}
        if isinstance(event, events.InputAudioBufferClearedEvent):
            return {"type": "input_audio_buffer.cleared"}
        if isinstance(event, events.ConversationItemInputAudioTranscriptionCompletedEvent):
            return {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": event.item_id,
                "transcript": event.transcript,
            }
        if isinstance(event, events.ConversationItemInputAudioTranscriptionFailedEvent):
            return {
                "type": "conversation.item.input_audio_transcription.failed",
                "item_id": event.item_id,
                "error": {"message": event.error_message},
            }
        if isinstance(event, events.ConversationItemTruncatedEvent):
            return {"type": "conversation.item.truncated", "item_id": event.item_id}
        if isinstance(event, events.SessionUpdatedEvent):
            return {
                "type": "session.updated",
                "session": {
                    "id": event.session_id,
                    "model": event.pipeline_name,
                    "modalities": {
                        "input": event.modalities_input,
                        "output": event.modalities_output,
                    },
                    "turn_detection": event.turn_detection,
                    "voice": event.voice,
                },
            }
        raise TypeError(f"No GA serialization implemented for event type {type(event)!r}")


class BetaDialectSerializer:
    name = "beta"

    def serialize(self, event: object) -> dict:
        if isinstance(event, events.SessionCreatedEvent):
            return {
                "type": "session.created",
                "session": {
                    "id": event.session_id,
                    "model": event.pipeline_name,
                    "modalities": event.modalities_output,
                    "turn_detection": event.turn_detection,
                    "voice": event.voice,
                },
            }
        if isinstance(event, events.ResponseCreatedEvent):
            return {
                "type": "response.created",
                "response": {"id": event.response_id, "status": "in_progress"},
            }
        if isinstance(event, events.ResponseOutputTextDeltaEvent):
            return {
                "type": "response.text.delta",
                "response_id": event.response_id,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
                "delta": event.delta,
            }
        if isinstance(event, events.ResponseOutputTextDoneEvent):
            return {
                "type": "response.text.done",
                "response_id": event.response_id,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
                "text": event.text,
            }
        if isinstance(event, events.ResponseOutputAudioDeltaEvent):
            return {
                "type": "response.audio.delta",
                "response_id": event.response_id,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
                "delta": base64.b64encode(event.delta_bytes).decode("ascii"),
            }
        if isinstance(event, events.ResponseOutputAudioDoneEvent):
            return {
                "type": "response.audio.done",
                "response_id": event.response_id,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
            }
        if isinstance(event, events.ResponseOutputAudioTranscriptDeltaEvent):
            return {
                "type": "response.audio_transcript.delta",
                "response_id": event.response_id,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
                "delta": event.delta,
            }
        if isinstance(event, events.ResponseOutputAudioTranscriptDoneEvent):
            return {
                "type": "response.audio_transcript.done",
                "response_id": event.response_id,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
                "transcript": event.transcript,
            }
        if isinstance(event, (events.ConversationItemAddedEvent, events.ConversationItemDoneEvent)):
            return {
                "type": "conversation.item.created",
                "item": {
                    "id": event.item_id,
                    "role": event.role,
                    "content": [{"type": "input_text", "text": event.text}],
                },
            }
        if isinstance(event, events.ResponseDoneEvent):
            return {
                "type": "response.done",
                "response": {"id": event.response_id, "status": event.status},
            }
        if isinstance(event, events.ErrorEvent):
            return {
                "type": "error",
                "error": {"code": event.code, "message": event.message},
                "event_id": event.event_id,
            }
        if isinstance(event, events.InputAudioBufferSpeechStartedEvent):
            return {"type": "input_audio_buffer.speech_started", "item_id": event.item_id}
        if isinstance(event, events.InputAudioBufferSpeechStoppedEvent):
            return {"type": "input_audio_buffer.speech_stopped", "item_id": event.item_id}
        if isinstance(event, events.InputAudioBufferCommittedEvent):
            return {"type": "input_audio_buffer.committed", "item_id": event.item_id}
        if isinstance(event, events.InputAudioBufferClearedEvent):
            return {"type": "input_audio_buffer.cleared"}
        if isinstance(event, events.ConversationItemInputAudioTranscriptionCompletedEvent):
            return {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": event.item_id,
                "transcript": event.transcript,
            }
        if isinstance(event, events.ConversationItemInputAudioTranscriptionFailedEvent):
            return {
                "type": "conversation.item.input_audio_transcription.failed",
                "item_id": event.item_id,
                "error": {"message": event.error_message},
            }
        if isinstance(event, events.ConversationItemTruncatedEvent):
            return {"type": "conversation.item.truncated", "item_id": event.item_id}
        if isinstance(event, events.SessionUpdatedEvent):
            return {
                "type": "session.updated",
                "session": {
                    "id": event.session_id,
                    "model": event.pipeline_name,
                    "modalities": event.modalities_output,
                    "turn_detection": event.turn_detection,
                    "voice": event.voice,
                },
            }
        raise TypeError(f"No beta serialization implemented for event type {type(event)!r}")


def select_dialect(openai_beta_header: str | None) -> GADialectSerializer | BetaDialectSerializer:
    """Dialect selection follows the client's OpenAI-Beta header --
    GA when absent (default), beta when present."""
    if openai_beta_header and "realtime=v1" in openai_beta_header:
        return BetaDialectSerializer()
    return GADialectSerializer()
