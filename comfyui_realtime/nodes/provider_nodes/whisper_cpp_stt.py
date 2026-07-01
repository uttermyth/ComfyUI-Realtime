from __future__ import annotations

import os

import folder_paths
from comfy_api.latest import io

from ..io_types import STTProviderType
from ...providers.whisper_cpp_stt import WhisperCppSTTProvider

folder_paths.add_model_folder_path(
    "whisper_cpp",
    os.path.join(folder_paths.models_dir, "stt", "whisper_cpp"),
)


class WhisperCppSTTProviderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WhisperCppSTTProviderNode",
            display_name="WhisperCpp STT Provider",
            category="Realtime/Providers",
            description=(
                "Speech-to-text using whisper.cpp. "
                "Place .bin model files in models/stt/whisper_cpp/. "
                "Connect the output to the stt input of Realtime Pipeline."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("whisper_cpp"),
                    tooltip="Whisper .bin model file from models/stt/whisper_cpp/.",
                ),
            ],
            outputs=[STTProviderType.Output(display_name="stt")],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        model_path = folder_paths.get_full_path_or_raise("whisper_cpp", model_name)
        return io.NodeOutput(WhisperCppSTTProvider(model_path=model_path))
