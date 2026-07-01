"""Global registry of active realtime sessions.

Mirrors registry.py's PipelineRegistry thread-safety pattern: a plain
threading.Lock, even though sessions are created and read from the same
aiohttp event-loop thread today -- consistent with PipelineRegistry's own
documented rationale (cross-thread registration is a real concern
elsewhere in this codebase, e.g. provider nodes execute on ComfyUI's
prompt-executor thread; staying consistent costs nothing here).

This registry exists for two purposes only: REST observability
(/realtime/sessions, /realtime/health's active_sessions count) and as a
safety gate for PipelineRegistry's reference-counted unload (registry.py)
-- it is NOT a second source of truth for anything the protocol logic
itself depends on. websocket_handler.py's own SessionState fields remain
the only thing the engine acts on.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .engine.session_state import SessionState


@dataclass
class SessionRecord:
    session_id: str
    pipeline_name: str
    connected_at: float
    state: SessionState


class SessionRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionRecord] = {}

    def register(self, session_id: str, pipeline_name: str, state: SessionState) -> None:
        with self._lock:
            self._sessions[session_id] = SessionRecord(
                session_id=session_id,
                pipeline_name=pipeline_name,
                connected_at=time.monotonic(),
                state=state,
            )

    def unregister(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def list(self) -> list[SessionRecord]:
        with self._lock:
            return list(self._sessions.values())

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


def derive_session_status(record: SessionRecord) -> str:
    task = record.state.active_response_task
    if task is not None and not task.done():
        if task.cancelling() > 0:
            return "cancelling"
        return "generating"
    if record.state.in_speech:
        return "listening"
    return "idle"


# Process-wide singleton, mirroring registry.py's pipeline_registry.
session_registry = SessionRegistry()
