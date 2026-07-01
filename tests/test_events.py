import re

from comfyui_realtime.engine import events


def test_new_id_has_prefix_and_is_unique():
    a = events.new_id("sess")
    b = events.new_id("sess")
    assert re.match(r"^sess_[0-9a-f]{24}$", a)
    assert a != b


def test_session_created_event_fields():
    event = events.SessionCreatedEvent(
        session_id="sess_x",
        pipeline_name="echo",
        modalities_input=["text"],
        modalities_output=["text"],
    )
    assert event.session_id == "sess_x"
    assert event.modalities_output == ["text"]


def test_session_created_event_turn_detection_defaults_to_none():
    event = events.SessionCreatedEvent(
        session_id="sess_x", pipeline_name="echo", modalities_input=["text"], modalities_output=["text"],
    )
    assert event.turn_detection is None


def test_session_created_event_voice_defaults_to_none():
    event = events.SessionCreatedEvent(
        session_id="sess_x", pipeline_name="echo", modalities_input=["text"], modalities_output=["text"],
    )
    assert event.voice is None


def test_session_updated_event_fields():
    event = events.SessionUpdatedEvent(
        session_id="sess_x", pipeline_name="echo", modalities_input=["text"], modalities_output=["text"],
    )
    assert event.session_id == "sess_x"
    assert event.turn_detection is None


def test_response_output_text_delta_event_fields():
    event = events.ResponseOutputTextDeltaEvent(
        response_id="resp_x",
        item_id="item_x",
        output_index=0,
        content_index=0,
        delta="hi",
    )
    assert event.delta == "hi"
