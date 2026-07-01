"""Shared pytest fixtures. COMFYUI_ROOT/COMFYUI_URL are used only by
tests marked `integration`, which require a live ComfyUI server."""
import asyncio
import os
import pathlib
import sys

COMFYUI_ROOT = pathlib.Path(
    os.environ.get(
        "COMFYUI_ROOT",
        "/Users/kale/Development/comfyui_installs/ComfyUI_v0.22.0_comfyui-realtime-dev",
    )
)
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")


def discover_v3_node_classes() -> dict:
    """Call __init__.py's comfy_entrypoint() the same way ComfyUI's real
    loader does (nodes.py:load_custom_node), and return {node_id:
    node_class} for every node the extension exposes -- the v3 replacement
    for reading NODE_CLASS_MAPPINGS as a static module attribute (comfy-org
    v3 migration). Both comfy_entrypoint() and get_node_list() are async;
    this is called from plain (non-async) test functions, so each call
    runs its own short-lived event loop via asyncio.run() -- safe here
    since neither does any real I/O, just object construction and a
    static list return.
    """
    init_module = sys.modules["__init__"]
    extension = asyncio.run(init_module.comfy_entrypoint())
    node_classes = asyncio.run(extension.get_node_list())
    return {node_cls.define_schema().node_id: node_cls for node_cls in node_classes}
