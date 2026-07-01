"""Pipeline and provider stub nodes, using ComfyUI's v3 node API.

EchoLLMProviderNode always returns the same fixed string -- a lightweight,
model-free smoke-test for validating the registration -> websocket round
trip alongside real providers like LlamaCppLLMProvider.

RealtimePipelineNode accepts LLM and optional TTS/voice/instructions/
temperature/VAD/STT inputs.
"""
from __future__ import annotations

import logging

from comfy_api.latest import io

from .io_types import LLMProviderType, STTProviderType, TTSProviderType, VADProviderType
from ..registry import PipelineConfig, pipeline_registry
from ..providers.base import GenerationDelta

logger = logging.getLogger("comfyui_realtime")


class _FixedStringLLM:
    def __init__(self, fixed_response: str) -> None:
        self._fixed_response = fixed_response

    async def generate(self, messages, options):
        yield GenerationDelta(text=self._fixed_response, finished=False)
        yield GenerationDelta(text="", finished=True)

    def unload(self) -> None:
        pass


class EchoLLMProviderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EchoLLMProviderNode",
            display_name="Echo LLM Provider (dev stub)",
            category="Realtime/Providers",
            inputs=[io.String.Input("fixed_response", default="You said: ")],
            outputs=[LLMProviderType.Output(display_name="llm")],
        )

    @classmethod
    def execute(cls, fixed_response) -> io.NodeOutput:
        return io.NodeOutput(_FixedStringLLM(fixed_response))


class RealtimePipelineNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RealtimePipelineNode",
            display_name="Realtime Pipeline",
            category="Realtime",
            is_output_node=True,
            description=(
                "Wires STT, VAD, LLM, and TTS providers into a named realtime voice pipeline. "
                "This is the output node — connect your providers here and give the pipeline a name. "
                "The realtime frontend selects a pipeline by that name."
            ),
            inputs=[
                io.String.Input(
                    "pipeline_name",
                    default="echo",
                    tooltip="Unique name for this pipeline, referenced by the realtime frontend to select which pipeline session to use.",
                ),
                LLMProviderType.Input(
                    "llm",
                    optional=True,
                    tooltip="Language model provider that generates text responses.",
                ),
                VADProviderType.Input(
                    "vad",
                    optional=True,
                    tooltip="Voice activity detection provider. Optional — if omitted, all audio is passed directly to STT.",
                ),
                STTProviderType.Input(
                    "stt",
                    optional=True,
                    tooltip="Speech-to-text provider for transcribing microphone input. Required if a VAD provider is connected.",
                ),
                TTSProviderType.Input(
                    "tts",
                    optional=True,
                    tooltip="Text-to-speech provider that converts LLM responses to audio.",
                ),
                io.String.Input(
                    "voice",
                    default="",
                    optional=True,
                    tooltip="Voice ID passed to the TTS provider at runtime. Must match an ID registered in the TTS provider node.",
                ),
                io.String.Input(
                    "instructions",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="System prompt injected at the start of each conversation turn. Leave blank to use the provider's default.",
                ),
                io.Float.Input(
                    "temperature",
                    default=0.8,
                    min=0.0,
                    max=2.0,
                    optional=True,
                    tooltip="Sampling temperature for LLM text generation. Higher values produce more varied output; lower values are more deterministic.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls, pipeline_name, llm=None, vad=None, stt=None, tts=None, voice="", instructions="", temperature=0.8
    ) -> io.NodeOutput:
        if llm is None and vad is None and stt is None and tts is None:
            raise ValueError(
                f"Pipeline '{pipeline_name}' has no providers configured (no llm/vad/stt/tts) -- "
                "every pipeline needs at least one productive stage."
            )
        if vad is not None and stt is None:
            logger.warning(
                "Pipeline '%s' has a VAD provider but no STT provider -- VAD will never be "
                "consulted, since audio input is rejected at the STT check before VAD runs. "
                "This combination is valid but incoherent.",
                pipeline_name,
            )
        pipeline_registry.register(
            PipelineConfig(
                name=pipeline_name,
                llm=llm,
                vad=vad,
                stt=stt,
                tts=tts,
                voice=voice or None,
                instructions=instructions or None,
                temperature=temperature,
            )
        )
        return io.NodeOutput(ui={"status": [f"Pipeline '{pipeline_name}' registered"]})

    @classmethod
    def fingerprint_inputs(
        cls, pipeline_name, llm=None, vad=None, stt=None, tts=None, voice="", instructions="", temperature=0.8
    ):
        # Always re-run registration, even when ComfyUI's node-output cache
        # would otherwise skip this node. fingerprint_inputs is v3's rename
        # of v1's IS_CHANGED -- semantics unchanged: returning a never-equal
        # value (NaN != NaN) forces a cache miss every time.
        return float("nan")
