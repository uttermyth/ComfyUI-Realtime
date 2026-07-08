from __future__ import annotations

import os

import folder_paths
from comfy_api.latest import io

from ..io_types import LLMProviderType
from ...providers.vllm_llm import VLLMProvider
from ._hf_model_dirs import _list_transformer_model_dirs


class VLLMProviderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VLLMProviderNode",
            display_name="vLLM LLM Provider",
            category="Realtime/Providers",
            description=(
                "Loads a HuggingFace transformers-format chat model (config.json + "
                ".safetensors weights + tokenizer files) via vLLM's synchronous "
                "LLMEngine, running fully in-process (single GPU only, "
                "tensor_parallel_size=1). "
                "Place model dirs in models/llm/transformers/"
                "Requires a CUDA GPU and a tokenizer with a chat_template."
                "Quantized checkpoints (NVFP4/modelopt, FP8, AWQ, GPTQ, ...) "
                "are auto-detected from the checkpoint's own config.json. "
                "Concurrent realtime sessions sharing this node's pipeline "
                "are serialized (one generation in flight at a time), same as "
                "the llama.cpp and transformers LLM providers."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=_list_transformer_model_dirs(),
                    tooltip="Model directory to load, scanned from models/llm/transformers/",
                ),
                io.Float.Input(
                    "gpu_memory_utilization",
                    default=0.9,
                    min=0.05,
                    max=1.0,
                    tooltip="Fraction of GPU memory vLLM is allowed to reserve for weights, "
                    "activations, and the KV cache.",
                ),
                io.Int.Input(
                    "max_model_len",
                    default=0,
                    min=0,
                    max=1048576,
                    tooltip="Maximum context length in tokens. 0 means use the model's own "
                    "default from its config.json.",
                ),
                io.Combo.Input(
                    "dtype",
                    options=["auto", "float16", "bfloat16", "float32"],
                    default="auto",
                    tooltip="Weight/activation precision. 'auto' lets vLLM pick based on the "
                    "checkpoint.",
                ),
                io.String.Input(
                    "quantization",
                    default="",
                    tooltip="Force a specific vLLM quantization method name (e.g. 'modelopt', "
                    "'fp8', 'awq'). Leave blank to auto-detect from the checkpoint's own "
                    "config.json -- this is how NVFP4/FP8/AWQ/GPTQ checkpoints all load "
                    "correctly without any extra configuration.",
                ),
                io.Boolean.Input(
                    "enforce_eager",
                    default=False,
                    tooltip="Disable CUDA graph capture. Slower steady-state generation, but "
                    "faster/simpler engine startup -- useful for debugging.",
                ),
                io.Boolean.Input(
                    "trust_remote_code",
                    default=False,
                    tooltip="Allow loading custom model code shipped in the model directory. "
                    "Only enable this for models you trust -- it executes local Python code "
                    "from the model directory.",
                ),
                io.String.Input(
                    "engine_args",
                    multiline=True,
                    default="",
                    tooltip="Optional JSON object overlaid onto vLLM's EngineArgs for "
                    "anything not exposed above (e.g. {\"tensor_parallel_size\": 2}). Unknown "
                    "keys raise an error naming the closest valid field.",
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
    def execute(
        cls,
        model_name,
        gpu_memory_utilization,
        max_model_len,
        dtype,
        quantization,
        enforce_eager,
        trust_remote_code,
        engine_args,
        system_prompt,
    ) -> io.NodeOutput:
        for base_dir in folder_paths.get_folder_paths("llm-transformers"):
            candidate = os.path.join(base_dir, model_name)
            if os.path.isdir(candidate):
                model_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Model directory {model_name!r} not found under any registered "
                f"llm-transformers path: {folder_paths.get_folder_paths('llm-transformers')}"
            )
        provider = VLLMProvider(
            model_path=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len or None,
            dtype=dtype,
            quantization=quantization,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            engine_args=engine_args,
            system_prompt=system_prompt,
        )
        return io.NodeOutput(provider)
