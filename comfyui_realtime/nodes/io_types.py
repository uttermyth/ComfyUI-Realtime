# comfyui_realtime/nodes/io_types.py
"""Custom ComfyUI v3 node-graph types for this project's providers.
Declared once, here, and imported by every node-defining module that
produces or consumes one -- io.Custom(...) instances must be shared, not
redeclared per file, so ComfyUI's graph type-checking treats every
LLM_PROVIDER (etc.) reference across the whole package as the same type."""
from __future__ import annotations

from comfy_api.latest import io

LLMProviderType = io.Custom("LLM_PROVIDER")
VADProviderType = io.Custom("VAD_PROVIDER")
STTProviderType = io.Custom("STT_PROVIDER")
TTSProviderType = io.Custom("TTS_PROVIDER")
