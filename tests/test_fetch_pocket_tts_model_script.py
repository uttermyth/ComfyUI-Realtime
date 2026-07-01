"""Requires pocket-tts installed. Only exercises URL-computation logic --
no network calls, no folder_paths dependency (main()'s actual download
logic imports folder_paths lazily inside main(), not at module level)."""
import importlib.util
import pathlib

import pytest

pytest.importorskip("pocket_tts")

_SCRIPT_PATH = pathlib.Path(__file__).parent.parent / "scripts" / "fetch_pocket_tts_model.py"
_spec = importlib.util.spec_from_file_location("fetch_pocket_tts_model", _SCRIPT_PATH)
fetch_pocket_tts_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_pocket_tts_model)


def test_urls_for_language_returns_tokenizer_and_weights_urls():
    urls = fetch_pocket_tts_model._urls_for_language("english")
    assert set(urls.keys()) == {"tokenizer.model", "model_without_voice_cloning.safetensors"}
    for url in urls.values():
        assert url.startswith(
            "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/"
        )
