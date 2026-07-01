import asyncio

from comfyui_realtime.engine.session_state import SessionState
from comfyui_realtime.session_registry import SessionRegistry, derive_session_status


def test_register_and_list():
    registry = SessionRegistry()
    state = SessionState(session_id="sess_x")
    registry.register("sess_x", "echo", state)
    records = registry.list()
    assert len(records) == 1
    assert records[0].session_id == "sess_x"
    assert records[0].pipeline_name == "echo"
    assert records[0].state is state


def test_unregister():
    registry = SessionRegistry()
    registry.register("sess_x", "echo", SessionState())
    assert registry.count() == 1
    registry.unregister("sess_x")
    assert registry.count() == 0
    registry.unregister("sess_x")  # unregistering twice is a safe no-op


def test_derive_session_status_idle_by_default():
    state = SessionState()
    record_list = SessionRegistry()
    record_list.register("sess_x", "echo", state)
    assert derive_session_status(record_list.list()[0]) == "idle"


def test_derive_session_status_listening():
    state = SessionState(in_speech=True)
    registry = SessionRegistry()
    registry.register("sess_x", "echo", state)
    assert derive_session_status(registry.list()[0]) == "listening"


async def test_derive_session_status_generating():
    state = SessionState()

    async def slow():
        await asyncio.sleep(10)

    state.active_response_task = asyncio.create_task(slow())
    registry = SessionRegistry()
    registry.register("sess_x", "echo", state)
    try:
        assert derive_session_status(registry.list()[0]) == "generating"
    finally:
        state.active_response_task.cancel()
        try:
            await state.active_response_task
        except asyncio.CancelledError:
            pass


async def test_derive_session_status_cancelling():
    state = SessionState()

    async def slow():
        await asyncio.sleep(10)

    state.active_response_task = asyncio.create_task(slow())
    registry = SessionRegistry()
    registry.register("sess_x", "echo", state)
    await asyncio.sleep(0)  # let the task actually start
    state.active_response_task.cancel()
    try:
        # Task.cancelling() > 0 once cancel() has been requested but before
        # the task has actually finished unwinding -- distinguishes
        # "being cancelled" from "generating" with zero new SessionState
        # fields (Python 3.13's asyncio.Task.cancelling()).
        assert derive_session_status(registry.list()[0]) == "cancelling"
    finally:
        try:
            await state.active_response_task
        except asyncio.CancelledError:
            pass
