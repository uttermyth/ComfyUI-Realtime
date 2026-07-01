from __future__ import annotations

import os

import folder_paths
from comfy_api.latest import io

from ..io_types import STTProviderType
from ...providers.faster_whisper_stt import FasterWhisperSTTProvider

folder_paths.add_model_folder_path(
    "faster_whisper",
    os.path.join(folder_paths.models_dir, "stt", "faster_whisper"),
)


def _list_faster_whisper_models() -> list[str]:
    """List CTranslate2 model directories from all registered faster_whisper paths."""
    models = []
    for base_dir in folder_paths.get_folder_paths("faster_whisper"):
        if os.path.isdir(base_dir):
            for entry in os.scandir(base_dir):
                if entry.is_dir():
                    models.append(entry.name)
    return sorted(models)


def _resolve_faster_whisper_path(model_name: str) -> str:
    """Return absolute path if model_name is a directory in a registered path, else return as-is."""
    for base_dir in folder_paths.get_folder_paths("faster_whisper"):
        candidate = os.path.join(base_dir, model_name)
        if os.path.isdir(candidate):
            return candidate
    return model_name


class FasterWhisperSTTProviderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FasterWhisperSTTProviderNode",
            display_name="Faster Whisper STT Provider",
            category="Realtime/Providers",
            description=(
                "Speech-to-text using faster-whisper (CTranslate2-optimized Whisper). "
                "Place CTranslate2-format model directories in models/stt/faster_whisper/. "
                "Connect the output to the stt input of Realtime Pipeline."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=_list_faster_whisper_models(),
                    tooltip="CTranslate2 model directory from models/stt/faster_whisper/. Convert Whisper checkpoints with the faster-whisper CLI, or download pre-converted models.",
                ),
            ],
            outputs=[STTProviderType.Output(display_name="stt")],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        model_path = _resolve_faster_whisper_path(model_name)
        return io.NodeOutput(FasterWhisperSTTProvider(model_path=model_path))
