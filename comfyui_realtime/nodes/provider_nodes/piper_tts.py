from __future__ import annotations

import json
import os

import folder_paths
from comfy_api.latest import io

from ..io_types import TTSProviderType
from ...providers.piper_tts import PiperTTSProvider

_CUSTOM_NODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

folder_paths.add_model_folder_path(
    "piper_voices",
    os.path.join(_CUSTOM_NODE_DIR, "assets", "piper_voices"),
)
folder_paths.add_model_folder_path(
    "piper_voices",
    os.path.join(folder_paths.models_dir, "tts", "piper_voices"),
)


def _list_piper_onnx_files() -> list[str]:
    """List .onnx voice files from all registered piper_voices paths."""
    files = []
    for base_dir in folder_paths.get_folder_paths("piper_voices"):
        for root, _, fnames in os.walk(base_dir):
            for fname in fnames:
                if fname.endswith(".onnx") and not fname.endswith(".onnx.json"):
                    files.append(os.path.relpath(os.path.join(root, fname), base_dir))
    return sorted(files)


def _list_piper_config_files() -> list[str]:
    """List .onnx.json config files from all registered piper_voices paths."""
    files = []
    for base_dir in folder_paths.get_folder_paths("piper_voices"):
        for root, _, fnames in os.walk(base_dir):
            for fname in fnames:
                if fname.endswith(".onnx.json"):
                    files.append(os.path.relpath(os.path.join(root, fname), base_dir))
    return sorted(files)


def _resolve_piper_voice_path(value: str) -> str:
    """Resolve a combo selection or annotated upload path to an absolute file path."""
    if value.endswith(("[input]", "[output]", "[temp]")):
        return folder_paths.get_annotated_filepath(value)
    if os.path.isabs(value):
        return value
    for base_dir in folder_paths.get_folder_paths("piper_voices"):
        candidate = os.path.join(base_dir, value)
        if os.path.isfile(candidate):
            return candidate
    return value


class PiperTTSProviderNode(io.ComfyNode):
    """Loads one or more Piper voices into a single TTS provider instance.
    The primary voice slot uses a file-browser combo for the onnx path; the
    config path is always <onnx_path>.json and is derived automatically.
    helper_onnx_path and helper_config_path are optional file pickers that
    let users discover paths to paste into additional_voices_json; they have
    no effect on the loaded provider."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        onnx_files = _list_piper_onnx_files()
        config_files = _list_piper_config_files()
        return io.Schema(
            node_id="PiperTTSProviderNode",
            display_name="Piper TTS Provider",
            category="Realtime/Providers",
            description=(
                "Text-to-speech using Piper, a fast local neural TTS engine. "
                "Supports multiple named voices in a single provider. "
                "Place .onnx voice files in models/tts/piper_voices/. "
                "Connect the output to the tts input of Realtime Pipeline."
            ),
            inputs=[
                io.Combo.Input(
                    "default_voice_path",
                    options=onnx_files,
                    upload=io.UploadType.model,
                    tooltip="Primary voice ONNX file, scanned from models/tts/piper_voices/. The matching .onnx.json config is loaded automatically from the same path.",
                ),
                io.String.Input(
                    "default_voice_id",
                    default="",
                    tooltip="Internal identifier for the default voice. Used to select this voice at runtime via the voice input of Realtime Pipeline.",
                ),
                io.String.Input(
                    "additional_voices_json",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip='JSON array of extra voice definitions. Each entry requires "id", "onnx_path", and "config_path" keys. Leave blank to use only the default voice.',
                ),
            ],
            outputs=[TTSProviderType.Output(display_name="tts")],
        )

    @classmethod
    def execute(
        cls,
        default_voice_path,
        default_voice_id,
        additional_voices_json="",
        # helper_onnx_path=None,
        # helper_config_path=None,
    ) -> io.NodeOutput:
        onnx_abs = _resolve_piper_voice_path(default_voice_path)
        config_abs = onnx_abs + ".json"
        voices = {default_voice_id: (onnx_abs, config_abs)}
        voices.update(cls._parse_additional_voices(additional_voices_json))
        provider = PiperTTSProvider(voices=voices, default_voice=default_voice_id)
        return io.NodeOutput(provider)

    @staticmethod
    def _parse_additional_voices(additional_voices_json: str) -> dict[str, tuple[str, str]]:
        if not additional_voices_json.strip():
            return {}
        try:
            entries = json.loads(additional_voices_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"additional_voices_json is not valid JSON: {exc}") from exc
        if not isinstance(entries, list):
            raise ValueError("additional_voices_json must be a JSON array of voice entries")
        voices: dict[str, tuple[str, str]] = {}
        for entry in entries:
            for required_key in ("id", "onnx_path", "config_path"):
                if required_key not in entry:
                    raise ValueError(f"additional_voices_json entry missing required key '{required_key}': {entry}")
            voices[entry["id"]] = (entry["onnx_path"], entry["config_path"])
        return voices
