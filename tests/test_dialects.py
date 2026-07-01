from comfyui_realtime.engine import dialects, events


def test_select_dialect_defaults_to_ga():
    dialect = dialects.select_dialect(None)
    assert dialect.name == "ga"


def test_select_dialect_returns_beta_for_beta_header():
    dialect = dialects.select_dialect("realtime=v1")
    assert dialect.name == "beta"


def test_ga_serializes_session_created():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.SessionCreatedEvent(
            session_id="sess_x",
            pipeline_name="echo",
            modalities_input=["text"],
            modalities_output=["text"],
        )
    )
    assert wire == {
        "type": "session.created",
        "session": {
            "id": "sess_x",
            "model": "echo",
            "modalities": {"input": ["text"], "output": ["text"]},
            "turn_detection": None,
            "voice": None,
        },
    }


def test_ga_session_created_reports_server_vad_turn_detection():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.SessionCreatedEvent(
            session_id="sess_x", pipeline_name="echo", modalities_input=["text", "audio"],
            modalities_output=["text"], turn_detection={"type": "server_vad"},
        )
    )
    assert wire["session"]["turn_detection"] == {"type": "server_vad"}


def test_ga_session_created_turn_detection_null_by_default():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.SessionCreatedEvent(
            session_id="sess_x", pipeline_name="echo", modalities_input=["text"], modalities_output=["text"],
        )
    )
    assert wire["session"]["turn_detection"] is None


def test_beta_session_created_reports_server_vad_turn_detection():
    dialect = dialects.BetaDialectSerializer()
    wire = dialect.serialize(
        events.SessionCreatedEvent(
            session_id="sess_x", pipeline_name="echo", modalities_input=["text", "audio"],
            modalities_output=["text"], turn_detection={"type": "server_vad"},
        )
    )
    assert wire["session"]["turn_detection"] == {"type": "server_vad"}


def test_ga_serializes_session_updated():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.SessionUpdatedEvent(
            session_id="sess_x", pipeline_name="echo", modalities_input=["text"], modalities_output=["text", "audio"],
            turn_detection={"type": "server_vad"},
        )
    )
    assert wire == {
        "type": "session.updated",
        "session": {
            "id": "sess_x",
            "model": "echo",
            "modalities": {"input": ["text"], "output": ["text", "audio"]},
            "turn_detection": {"type": "server_vad"},
            "voice": None,
        },
    }


def test_beta_serializes_session_updated():
    dialect = dialects.BetaDialectSerializer()
    wire = dialect.serialize(
        events.SessionUpdatedEvent(
            session_id="sess_x", pipeline_name="echo", modalities_input=["text"], modalities_output=["text", "audio"],
        )
    )
    assert wire["type"] == "session.updated"
    assert wire["session"]["modalities"] == ["text", "audio"]
    assert wire["session"]["turn_detection"] is None


def test_ga_session_created_includes_voice():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.SessionCreatedEvent(
            session_id="sess_x", pipeline_name="echo", modalities_input=["text"], modalities_output=["text"],
            voice="lessac-medium",
        )
    )
    assert wire["session"]["voice"] == "lessac-medium"


def test_ga_session_updated_includes_voice():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.SessionUpdatedEvent(
            session_id="sess_x", pipeline_name="echo", modalities_input=["text"], modalities_output=["text"],
            voice="lessac-medium-2",
        )
    )
    assert wire["session"]["voice"] == "lessac-medium-2"


def test_beta_session_created_includes_voice():
    dialect = dialects.BetaDialectSerializer()
    wire = dialect.serialize(
        events.SessionCreatedEvent(
            session_id="sess_x", pipeline_name="echo", modalities_input=["text"], modalities_output=["text"],
            voice="lessac-medium",
        )
    )
    assert wire["session"]["voice"] == "lessac-medium"


def test_beta_session_updated_includes_voice():
    dialect = dialects.BetaDialectSerializer()
    wire = dialect.serialize(
        events.SessionUpdatedEvent(
            session_id="sess_x", pipeline_name="echo", modalities_input=["text"], modalities_output=["text"],
            voice="lessac-medium-2",
        )
    )
    assert wire["session"]["voice"] == "lessac-medium-2"


def test_ga_serializes_response_output_text_delta():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.ResponseOutputTextDeltaEvent(
            response_id="resp_x", item_id="item_x", output_index=0, content_index=0, delta="hi",
        )
    )
    assert wire["type"] == "response.output_text.delta"
    assert wire["delta"] == "hi"


def test_ga_serializes_error_event():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(events.ErrorEvent(code="pipeline_not_found", message="nope"))
    assert wire == {
        "type": "error",
        "error": {"code": "pipeline_not_found", "message": "nope"},
        "event_id": None,
    }


def test_beta_serializes_response_output_text_delta_as_response_text_delta():
    dialect = dialects.BetaDialectSerializer()
    wire = dialect.serialize(
        events.ResponseOutputTextDeltaEvent(
            response_id="resp_x", item_id="item_x", output_index=0, content_index=0, delta="hi",
        )
    )
    assert wire["type"] == "response.text.delta"
    assert wire["delta"] == "hi"


def test_beta_serializes_response_output_audio_delta_as_response_audio_delta():
    import base64

    dialect = dialects.BetaDialectSerializer()
    wire = dialect.serialize(
        events.ResponseOutputAudioDeltaEvent(
            response_id="resp_x", item_id="item_x", output_index=0, content_index=0, delta_bytes=b"\x01\x02",
        )
    )
    assert wire["type"] == "response.audio.delta"
    assert wire["delta"] == base64.b64encode(b"\x01\x02").decode("ascii")


def test_beta_serializes_response_output_audio_transcript_delta_as_response_audio_transcript_delta():
    dialect = dialects.BetaDialectSerializer()
    wire = dialect.serialize(
        events.ResponseOutputAudioTranscriptDeltaEvent(
            response_id="resp_x", item_id="item_x", output_index=0, content_index=0, delta="hi",
        )
    )
    assert wire["type"] == "response.audio_transcript.delta"


def test_beta_serializes_both_conversation_item_events_as_conversation_item_created():
    dialect = dialects.BetaDialectSerializer()
    added = dialect.serialize(events.ConversationItemAddedEvent(item_id="item_x", role="user", text="hi"))
    done = dialect.serialize(events.ConversationItemDoneEvent(item_id="item_x", role="user", text="hi"))
    assert added["type"] == "conversation.item.created"
    assert done["type"] == "conversation.item.created"


def test_beta_serializes_session_created_with_combined_modalities():
    dialect = dialects.BetaDialectSerializer()
    wire = dialect.serialize(
        events.SessionCreatedEvent(
            session_id="sess_x", pipeline_name="echo", modalities_input=["text"], modalities_output=["text", "audio"],
        )
    )
    assert wire["type"] == "session.created"
    assert wire["session"]["modalities"] == ["text", "audio"]


def test_beta_serializes_response_done_and_error_same_shape_as_ga():
    dialect = dialects.BetaDialectSerializer()
    done = dialect.serialize(events.ResponseDoneEvent(response_id="resp_x"))
    assert done["type"] == "response.done"
    error = dialect.serialize(events.ErrorEvent(code="x", message="y"))
    assert error["type"] == "error"


def test_ga_serializes_conversation_item_added():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.ConversationItemAddedEvent(item_id="item_x", role="user", text="hello")
    )
    assert wire == {
        "type": "conversation.item.added",
        "item": {"id": "item_x", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
    }


def test_ga_serializes_conversation_item_done():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.ConversationItemDoneEvent(item_id="item_x", role="user", text="hello")
    )
    assert wire["type"] == "conversation.item.done"
    assert wire["item"]["id"] == "item_x"


def test_ga_serializes_response_output_audio_delta():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.ResponseOutputAudioDeltaEvent(
            response_id="resp_x", item_id="item_x", output_index=0, content_index=0, delta_bytes=b"\x01\x02",
        )
    )
    assert wire["type"] == "response.output_audio.delta"
    import base64
    assert wire["delta"] == base64.b64encode(b"\x01\x02").decode("ascii")


def test_ga_serializes_response_output_audio_done():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.ResponseOutputAudioDoneEvent(response_id="resp_x", item_id="item_x", output_index=0, content_index=0)
    )
    assert wire["type"] == "response.output_audio.done"


def test_ga_serializes_response_output_audio_transcript_delta():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.ResponseOutputAudioTranscriptDeltaEvent(
            response_id="resp_x", item_id="item_x", output_index=0, content_index=0, delta="hi",
        )
    )
    assert wire["type"] == "response.output_audio_transcript.delta"
    assert wire["delta"] == "hi"


def test_ga_serializes_response_output_audio_transcript_done():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.ResponseOutputAudioTranscriptDoneEvent(
            response_id="resp_x", item_id="item_x", output_index=0, content_index=0, transcript="hi there",
        )
    )
    assert wire["type"] == "response.output_audio_transcript.done"
    assert wire["transcript"] == "hi there"


def test_ga_serializes_input_audio_buffer_speech_started():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(events.InputAudioBufferSpeechStartedEvent(item_id="item_x"))
    assert wire == {"type": "input_audio_buffer.speech_started", "item_id": "item_x"}


def test_ga_serializes_input_audio_buffer_speech_stopped():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(events.InputAudioBufferSpeechStoppedEvent(item_id="item_x"))
    assert wire == {"type": "input_audio_buffer.speech_stopped", "item_id": "item_x"}


def test_ga_serializes_input_audio_buffer_committed():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(events.InputAudioBufferCommittedEvent(item_id="item_x"))
    assert wire == {"type": "input_audio_buffer.committed", "item_id": "item_x"}


def test_ga_serializes_input_audio_buffer_cleared():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(events.InputAudioBufferClearedEvent())
    assert wire == {"type": "input_audio_buffer.cleared"}


def test_ga_serializes_transcription_completed():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.ConversationItemInputAudioTranscriptionCompletedEvent(item_id="item_x", transcript="hello")
    )
    assert wire["type"] == "conversation.item.input_audio_transcription.completed"
    assert wire["transcript"] == "hello"


def test_ga_serializes_transcription_failed():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(
        events.ConversationItemInputAudioTranscriptionFailedEvent(item_id="item_x", error_message="boom")
    )
    assert wire["type"] == "conversation.item.input_audio_transcription.failed"
    assert wire["error"]["message"] == "boom"


def test_ga_serializes_conversation_item_truncated():
    dialect = dialects.GADialectSerializer()
    wire = dialect.serialize(events.ConversationItemTruncatedEvent(item_id="item_x"))
    assert wire == {"type": "conversation.item.truncated", "item_id": "item_x"}


def test_beta_serializes_conversation_item_truncated():
    dialect = dialects.BetaDialectSerializer()
    wire = dialect.serialize(events.ConversationItemTruncatedEvent(item_id="item_x"))
    assert wire == {"type": "conversation.item.truncated", "item_id": "item_x"}


def test_beta_serializes_all_six_new_events_without_raising():
    dialect = dialects.BetaDialectSerializer()
    dialect.serialize(events.InputAudioBufferSpeechStartedEvent(item_id="item_x"))
    dialect.serialize(events.InputAudioBufferSpeechStoppedEvent(item_id="item_x"))
    dialect.serialize(events.InputAudioBufferCommittedEvent(item_id="item_x"))
    dialect.serialize(events.InputAudioBufferClearedEvent())
    dialect.serialize(
        events.ConversationItemInputAudioTranscriptionCompletedEvent(item_id="item_x", transcript="hi")
    )
    dialect.serialize(
        events.ConversationItemInputAudioTranscriptionFailedEvent(item_id="item_x", error_message="x")
    )
