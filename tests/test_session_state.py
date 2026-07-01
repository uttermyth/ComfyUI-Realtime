from comfyui_realtime.engine.session_state import SessionState


def test_session_state_defaults():
    session = SessionState()
    assert session.conversation_history == []
    assert session.active_response_task is None


def test_session_state_audio_fields_default():
    session = SessionState()
    assert session.audio_framer is None
    assert session.utterance_audio_16k == b""
    assert session.in_speech is False
    assert session.pending_item_id is None


def test_session_state_last_cancelled_response_defaults_to_none():
    session = SessionState()
    assert session.last_cancelled_response is None


def test_session_state_session_id_defaults_to_none():
    session = SessionState()
    assert session.session_id is None
