# comfyui_realtime/nodes/dev_nodes.py
"""Development/test-only nodes -- not part of the public provider API.

BusyWorkNode stands in for a heavy image-generation workflow, blocking
ComfyUI's prompt-executor thread for a fixed duration with CPU work, to
verify the aiohttp event loop (and active websocket sessions) stay
responsive while it runs. A CPU busy-loop is a faithful stand-in here
because the risk under test -- the worker thread blocking the event
loop -- is independent of whether the blocking work is GPU image
generation or a CPU loop.
"""
from __future__ import annotations

import hashlib
import time

from comfy_api.latest import io


class BusyWorkNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BusyWorkNode",
            display_name="Busy Work (dev only)",
            category="Realtime/Dev",
            is_output_node=True,
            inputs=[io.Float.Input("duration_seconds", default=5.0, min=0.1, max=120.0)],
            outputs=[],
        )

    @classmethod
    def execute(cls, duration_seconds) -> io.NodeOutput:
        deadline = time.perf_counter() + duration_seconds
        payload = b"busywork"
        iterations = 0
        while time.perf_counter() < deadline:
            payload = hashlib.sha256(payload).digest()
            iterations += 1
        return io.NodeOutput(ui={"status": [f"BusyWorkNode ran {iterations} hash iterations"]})
