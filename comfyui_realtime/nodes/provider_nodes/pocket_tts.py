from __future__ import annotations

import json
import os
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from ..io_types import TTSProviderType
from ...providers.pocket_tts import PocketTTSProvider

_CUSTOM_NODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

folder_paths.add_model_folder_path(
    "pocket_tts_voices",
    os.path.join(_CUSTOM_NODE_DIR, "assets", "pocket_tts_voices"),
)
folder_paths.add_model_folder_path(
    "pocket_tts_voices",
    os.path.join(folder_paths.models_dir, "tts", "pocket_tts_voices"),
)

_LANGUAGES = [
    "english",
    "english_2026-01",
    "english_2026-04",
    "french_24l",
    "german",
    "german_24l",
    "italian",
    "italian_24l",
    "portuguese",
    "portuguese_24l",
    "spanish",
    "spanish_24l",
]


def _list_pocket_tts_voice_files() -> list[str]:
    """List local audio/safetensors voice files from all registered pocket_tts_voices paths."""
    files = []
    for base_dir in folder_paths.get_folder_paths("pocket_tts_voices"):
        for root, _, fnames in os.walk(base_dir):
            for fname in fnames:
                if fname.endswith((".wav", ".mp3", ".safetensors")):
                    files.append(os.path.relpath(os.path.join(root, fname), base_dir))
    return sorted(files)


def _resolve_pocket_tts_voice_path(value: str) -> str:
    """Resolve a combo selection or annotated upload path to an absolute filesystem path.
    Falls through unchanged for anything that isn't a local file -- e.g. a preset voice
    name or hf:// URL a power user pasted in via a converted string input."""
    if value.endswith(("[input]", "[output]", "[temp]")):
        return folder_paths.get_annotated_filepath(value)
    if os.path.isabs(value):
        return value
    for base_dir in folder_paths.get_folder_paths("pocket_tts_voices"):
        candidate = os.path.join(base_dir, value)
        if os.path.isfile(candidate):
            return candidate
    return value


class PocketTTSProviderNode(io.ComfyNode):
    """Loads one or more Pocket TTS voices into a single TTS provider instance,
    sharing one underlying model. The primary voice slot uses a file-browser combo
    for a local audio (.wav/.mp3) or exported (.safetensors) voice file.
    helper_voice_path is an optional file picker for discovering paths to paste
    into additional_voices_json; it has no effect on the loaded provider."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        voice_files = _list_pocket_tts_voice_files()
        return io.Schema(
            node_id="PocketTTSProviderNode",
            display_name="Pocket TTS Provider",
            category="Realtime/Providers",
            description=(
                "Text-to-speech using Pocket TTS (Kyutai), a small local neural TTS "
                "model that clones a voice from a short reference audio clip. "
                "Supports multiple named voices sharing one loaded model. "
                "Place .wav/.mp3 reference clips or exported .safetensors voice "
                "states in models/tts/pocket_tts_voices/. Model weights are read "
                "from models/tts/pocket_tts/<language>/ -- run "
                "scripts/fetch_pocket_tts_model.py <language> once to fetch them. "
                "Connect the output to the tts input of Realtime Pipeline."
            ),
            inputs=[
                io.Combo.Input(
                    "language",
                    options=_LANGUAGES,
                    default="english",
                    tooltip="Model language/config to load. Requires model.safetensors (or "
                    "model_without_voice_cloning.safetensors) and tokenizer.model under "
                    "models/tts/pocket_tts/<language>/ -- run scripts/fetch_pocket_tts_model.py "
                    "<language> once to fetch them.",
                ),
                io.Combo.Input(
                    "default_voice_path",
                    options=voice_files,
                    upload=io.UploadType.audio,
                    tooltip="Primary voice: a local reference audio clip (.wav/.mp3, ~5-30s of "
                    "clean speech -- requires the optional voice-cloning weights) or a "
                    "pre-exported .safetensors voice state (works with the default weights), "
                    "scanned from models/tts/pocket_tts_voices/.",
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
                    tooltip='JSON array of extra voice definitions. Each entry requires "id" and "voice_source" keys. Leave blank to use only the default voice.',
                ),
                io.Combo.Input(
                    "device",
                    options=["cpu", "cuda", "mps"],
                    default="cpu",
                    optional=True,
                    tooltip="Torch device to run the model on.",
                ),
                io.Boolean.Input(
                    "quantize",
                    default=False,
                    optional=True,
                    tooltip="Apply dynamic int8 quantization to reduce memory and improve CPU inference speed, per pocket-tts's own docs. No measurable quality impact.",
                ),
                io.Combo.Input(
                    "helper_voice_path",
                    options=voice_files,
                    upload=io.UploadType.audio,
                    optional=True,
                    tooltip="Browse-only: pick a file here to see its resolved path, then paste it into additional_voices_json. Has no effect on the loaded provider.",
                ),
            ],
            outputs=[TTSProviderType.Output(display_name="tts")],
        )

    @classmethod
    def execute(
        cls,
        language,
        default_voice_path,
        default_voice_id,
        additional_voices_json="",
        device="cpu",
        quantize=False,
        helper_voice_path=None,
    ) -> io.NodeOutput:
        model_dir = Path(folder_paths.models_dir) / "tts" / "pocket_tts" / language
        default_abs = _resolve_pocket_tts_voice_path(default_voice_path)
        voices = {default_voice_id: default_abs}
        voices.update(cls._parse_additional_voices(additional_voices_json))
        provider = PocketTTSProvider(
            voices=voices,
            default_voice=default_voice_id,
            model_dir=model_dir,
            language=language,
            device=device,
            quantize=quantize,
        )
        return io.NodeOutput(provider)

    @staticmethod
    def _parse_additional_voices(additional_voices_json: str) -> dict[str, str]:
        if not additional_voices_json.strip():
            return {}
        try:
            entries = json.loads(additional_voices_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"additional_voices_json is not valid JSON: {exc}") from exc
        if not isinstance(entries, list):
            raise ValueError("additional_voices_json must be a JSON array of voice entries")
        voices: dict[str, str] = {}
        for entry in entries:
            for required_key in ("id", "voice_source"):
                if required_key not in entry:
                    raise ValueError(f"additional_voices_json entry missing required key '{required_key}': {entry}")
            voices[entry["id"]] = entry["voice_source"]
        return voices
