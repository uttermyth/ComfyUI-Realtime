from __future__ import annotations

import os

import folder_paths
from comfy_api.latest import io

from ..io_types import LLMProviderType
from ...providers.transformers_llm import TransformersLLMProvider

folder_paths.add_model_folder_path(
    "llm-transformers",
    os.path.join(folder_paths.models_dir, "llm", "transformers"),
)


def _list_transformer_model_dirs() -> list[str]:
    """List immediate subdirectories of every registered llm-transformers
    base path that contain a config.json -- the directory-based equivalent
    of LlamaCppLLMProviderNode's flat .gguf file listing (HF transformers
    models are directories, not single files)."""
    names = []
    for base_dir in folder_paths.get_folder_paths("llm-transformers"):
        if not os.path.isdir(base_dir):
            continue
        for entry in sorted(os.listdir(base_dir)):
            entry_path = os.path.join(base_dir, entry)
            if os.path.isdir(entry_path) and os.path.isfile(os.path.join(entry_path, "config.json")):
                names.append(entry)
    return names


class TransformersLLMProviderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TransformersLLMProviderNode",
            display_name="Transformers LLM Provider",
            category="Realtime/Providers",
            description=(
                "Loads a HuggingFace transformers-format chat model (config.json + "
                ".safetensors weights + tokenizer files) from a local directory. "
                "Place model directories in ComfyUI's models/llm/transformers/ "
                "directory. Requires a tokenizer with a chat_template -- "
                "base/completion-only models aren't supported. "
                "FP8-quantized checkpoints (quant_method: \"fp8\") are supported "
                "automatically via the checkpoint's own config.json, but require a "
                "GPU with Compute Capability >= 9 (Hopper H100 or newer, or "
                "Blackwell) -- no separate setting needed here. "
                "Connect the output to the llm input of Realtime Pipeline."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=_list_transformer_model_dirs(),
                    tooltip="Model directory to load, scanned from models/llm/transformers/.",
                ),
                io.Combo.Input(
                    "device",
                    options=["auto", "cuda", "mps", "cpu"],
                    default="auto",
                    tooltip="Torch device to run the model on. 'auto' picks cuda, then mps, then cpu.",
                ),
                io.Combo.Input(
                    "torch_dtype",
                    options=["auto", "float16", "bfloat16", "float32"],
                    default="auto",
                    tooltip="Weight precision. 'auto' picks float16 on cuda/mps, float32 on cpu.",
                ),
                io.Boolean.Input(
                    "trust_remote_code",
                    default=False,
                    tooltip="Allow loading custom model code shipped in the model directory. "
                    "Only enable this for models you trust -- it executes local Python code "
                    "from the model directory.",
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
    def execute(cls, model_name, device, torch_dtype, trust_remote_code, system_prompt) -> io.NodeOutput:
        for base_dir in folder_paths.get_folder_paths("llm-transformers"):
            candidate = os.path.join(base_dir, model_name)
            if os.path.isdir(candidate):
                model_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Transformers model directory {model_name!r} not found under any "
                f"registered llm-transformers path: {folder_paths.get_folder_paths('llm-transformers')}"
            )
        provider = TransformersLLMProvider(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            system_prompt=system_prompt,
        )
        return io.NodeOutput(provider)
