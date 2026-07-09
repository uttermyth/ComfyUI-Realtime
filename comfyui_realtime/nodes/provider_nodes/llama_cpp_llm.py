from __future__ import annotations

import os

import folder_paths
from comfy_api.latest import io

from ..io_types import LLMProviderType
from ...providers.llama_cpp_llm import LlamaCppLLMProvider

folder_paths.add_model_folder_path(
    "llm-gguf",
    os.path.join(folder_paths.models_dir, "llm", "gguf"),
)


class LlamaCppLLMProviderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppLLMProviderNode",
            display_name="LlamaCpp LLM Provider",
            category="Realtime/Providers",
            description=(
                "Loads a GGUF-format language model via llama.cpp. "
                "Place .gguf model files in ComfyUI's models/llm/ directory. "
                "Connect the output to the llm input of Realtime Pipeline."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=[f for f in folder_paths.get_filename_list("llm-gguf") if f.endswith(".gguf")],
                    tooltip="GGUF model file to load, scanned from models/llm/.",
                ),
                io.Int.Input(
                    "n_ctx",
                    default=4096,
                    min=64,
                    max=131072,
                    tooltip="Context window size in tokens. Larger values allow longer conversations but consume more memory. Typical range: 2048–32768.",
                ),
                io.Int.Input(
                    "n_gpu_layers",
                    default=-1,
                    min=-1,
                    max=200,
                    tooltip="Number of transformer layers to offload to GPU. Use -1 to offload all layers. Set to 0 to run entirely on CPU.",
                ),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default="",
                    tooltip="Instructions prepended to every conversation as the system message. Leave blank to use no system prompt.",
                ),
            ],
            outputs=[LLMProviderType.Output(display_name="llm")],
        )

    @classmethod
    def execute(cls, model_name, n_ctx, n_gpu_layers, system_prompt) -> io.NodeOutput:
        model_path = folder_paths.get_full_path_or_raise("llm", model_name)
        provider = LlamaCppLLMProvider(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            system_prompt=system_prompt,
        )
        return io.NodeOutput(provider)
