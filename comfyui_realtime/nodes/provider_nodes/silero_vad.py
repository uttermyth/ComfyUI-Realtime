from __future__ import annotations

from comfy_api.latest import io

from ..io_types import VADProviderType
from ...providers.silero_vad import SileroVADProvider


class SileroVADProviderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SileroVADProviderNode",
            display_name="Silero VAD Provider",
            category="Realtime/Providers",
            description=(
                "Voice activity detection using the Silero VAD model. "
                "Filters audio so STT only processes frames containing speech. "
                "Connect the output to the vad input of Realtime Pipeline alongside an STT provider."
            ),
            inputs=[
                io.Float.Input(
                    "threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    tooltip="Speech probability threshold (0.0–1.0). Frames above this confidence score are forwarded to STT. Lower values catch quieter speech; higher values reduce false positives.",
                ),
            ],
            outputs=[VADProviderType.Output(display_name="vad")],
        )

    @classmethod
    def execute(cls, threshold) -> io.NodeOutput:
        return io.NodeOutput(SileroVADProvider(threshold=threshold))
