"""PocketTTSProvider wraps pocket-tts (kyutai-labs/pocket-tts, MIT) -- a single
shared torch model conditioned per-voice by an audio-prompt state, unlike
Piper's per-voice ONNX models.

Voices are resolved to local audio/safetensors files by the node layer before
reaching this provider. The base model itself is loaded from local files too
(see _build_local_model_config below) -- this provider never makes a network
call of its own; if the required local model files are missing, construction
raises FileNotFoundError with exact download instructions instead of falling
back to fetching anything.

generate_audio_stream() yields float32 PCM chunks progressively (unlike
Piper's per-sentence-only granularity) -- each chunk is converted to int16
PCM and resampled to the wire rate via engine/resample.py before being
yielded.
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import AsyncIterator

import torch
import yaml
from pocket_tts import TTSModel
from pocket_tts.utils.config import CONFIGS_DIR

from ..engine.executor_bridge import bridge_sync_iterator
from ..engine.resample import resample_pcm16
from .base import VoiceInfo

_MODEL_FILENAME = "model.safetensors"
_MODEL_WITHOUT_VOICE_CLONING_FILENAME = "model_without_voice_cloning.safetensors"
_TOKENIZER_FILENAME = "tokenizer.model"


def _hf_pointer_to_https(pointer: str) -> str:
    """Convert pocket-tts's internal hf://<repo>/<path>@<revision> pointer format
    to a plain, public, unauthenticated HTTPS download URL."""
    repo_and_path, _, revision = pointer.removeprefix("hf://").partition("@")
    parts = repo_and_path.split("/")
    repo_id = "/".join(parts[:2])
    file_path = "/".join(parts[2:])
    return f"https://huggingface.co/{repo_id}/resolve/{revision or 'main'}/{file_path}"


def _missing_local_model_files_error(language: str, model_dir: Path) -> FileNotFoundError:
    with open(CONFIGS_DIR / f"{language}.yaml") as f:
        upstream = yaml.safe_load(f)
    tokenizer_url = _hf_pointer_to_https(upstream["flow_lm"]["lookup_table"]["tokenizer_path"])
    weights_url = _hf_pointer_to_https(upstream["weights_path_without_voice_cloning"])
    return FileNotFoundError(
        f"Pocket TTS model files for language '{language}' were not found under {model_dir}.\n"
        f"Download these two files (no Hugging Face account needed) and save them exactly as named:\n"
        f"  {model_dir / _TOKENIZER_FILENAME}\n    <- {tokenizer_url}\n"
        f"  {model_dir / _MODEL_WITHOUT_VOICE_CLONING_FILENAME}\n    <- {weights_url}\n"
        f"Or run: python scripts/fetch_pocket_tts_model.py {language}\n"
        f"For voice cloning from your own audio (optional -- requires accepting terms at "
        f"https://huggingface.co/kyutai/pocket-tts and downloading while logged in), save "
        f"that file instead as {model_dir / _MODEL_FILENAME}."
    )


def _build_local_model_config(language: str, model_dir: Path) -> tuple[Path, bool]:
    """Patch pocket-tts's bundled per-language config to point at local model
    files instead of its default hf:// pointers, and write the result to a
    temp YAML file (the caller loads it, then deletes it).

    Returns (temp_config_path, used_voice_cloning_weights). used_voice_cloning_weights
    is False when only the public, non-cloning weights file was found -- the caller
    must then set the loaded model's has_voice_cloning=False itself, since pointing
    weights_path directly at a local file bypasses the library's own try/except-based
    detection of which weights actually loaded (that detection only fires when a
    hf:// download raises, which never happens for a plain local path).
    """
    tokenizer_path = model_dir / _TOKENIZER_FILENAME
    model_path = model_dir / _MODEL_FILENAME
    no_clone_path = model_dir / _MODEL_WITHOUT_VOICE_CLONING_FILENAME

    if not tokenizer_path.is_file() or not (model_path.is_file() or no_clone_path.is_file()):
        raise _missing_local_model_files_error(language, model_dir)

    with open(CONFIGS_DIR / f"{language}.yaml") as f:
        config_dict = yaml.safe_load(f)

    config_dict["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer_path)
    if model_path.is_file():
        config_dict["weights_path"] = str(model_path)
        used_voice_cloning_weights = True
    else:
        config_dict["weights_path"] = str(no_clone_path)
        used_voice_cloning_weights = False

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    try:
        yaml.safe_dump(config_dict, tmp)
    finally:
        tmp.close()
    return Path(tmp.name), used_voice_cloning_weights


class PocketTTSProvider:
    output_format = "pcm16"

    def __init__(
        self,
        voices: dict[str, str],
        default_voice: str,
        model_dir: str | Path,
        language: str = "english",
        device: str = "cpu",
        quantize: bool = False,
    ) -> None:
        temp_config_path, used_voice_cloning_weights = _build_local_model_config(language, Path(model_dir))
        try:
            self._model = TTSModel.load_model(config=str(temp_config_path), quantize=quantize)
        finally:
            temp_config_path.unlink(missing_ok=True)
        if not used_voice_cloning_weights:
            self._model.has_voice_cloning = False
        self._model.to(device)
        self._model.eval()

        self._voice_states: dict[str, dict] = {
            voice_id: self._model.get_state_for_audio_prompt(source)
            for voice_id, source in voices.items()
        }
        self._default_voice = default_voice
        self._lock = threading.Lock()
        self.output_sample_rate = 24000

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id=voice_id, name=voice_id) for voice_id in self._voice_states]

    async def synthesize(
        self, text_stream: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[bytes]:
        voice_state = self._voice_states[voice or self._default_voice]
        native_rate = self._model.sample_rate

        self._lock.acquire()
        try:
            async for text in text_stream:
                stop_event = threading.Event()

                def factory(text: str = text):
                    for chunk in self._model.generate_audio_stream(voice_state, text):
                        pcm16 = chunk.clamp(-1, 1).mul(32767).to(torch.int16)
                        yield pcm16.cpu().numpy().tobytes()

                bridge = bridge_sync_iterator(factory, stop_event)
                try:
                    async for raw_chunk in bridge:
                        yield resample_pcm16(raw_chunk, native_rate, self.output_sample_rate)
                finally:
                    await bridge.aclose()
        finally:
            self._lock.release()

    def unload(self) -> None:
        self._voice_states.clear()
        self._model = None
