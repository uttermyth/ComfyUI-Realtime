#!/usr/bin/env python3
"""Download Pocket TTS's local model files for one language -- plain HTTPS,
no Hugging Face account or CLI required.

Usage: python scripts/fetch_pocket_tts_model.py english
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from pocket_tts.utils.config import CONFIGS_DIR

from comfyui_realtime.providers.pocket_tts import (
    _MODEL_WITHOUT_VOICE_CLONING_FILENAME,
    _TOKENIZER_FILENAME,
    _hf_pointer_to_https,
)


def _urls_for_language(language: str) -> dict[str, str]:
    with open(CONFIGS_DIR / f"{language}.yaml") as f:
        config = yaml.safe_load(f)
    return {
        _TOKENIZER_FILENAME: _hf_pointer_to_https(config["flow_lm"]["lookup_table"]["tokenizer_path"]),
        _MODEL_WITHOUT_VOICE_CLONING_FILENAME: _hf_pointer_to_https(config["weights_path_without_voice_cloning"]),
    }


def main() -> None:
    import folder_paths

    language = sys.argv[1] if len(sys.argv) > 1 else "english"
    model_dir = Path(folder_paths.models_dir) / "tts" / "pocket_tts" / language
    model_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in _urls_for_language(language).items():
        dest = model_dir / filename
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    print(f"Done. Set PocketTTSProviderNode's language input to '{language}'.")


if __name__ == "__main__":
    main()
