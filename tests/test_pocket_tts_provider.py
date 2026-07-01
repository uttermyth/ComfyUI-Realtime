"""Requires pocket-tts installed. The tests in this file don't need the real
~600MB model weights -- only tests marked pytest.mark.integration do (added
in Task 3). Those require model_without_voice_cloning.safetensors and
tokenizer.model under models/tts/pocket_tts/english/, and a voice embedding
at assets/pocket_tts_voices/sample_voice.safetensors -- see the plan's
"Prerequisite" section for exact download commands."""
import pathlib
import time

import pytest

pytest.importorskip("pocket_tts")

from comfyui_realtime.providers.pocket_tts import (
    PocketTTSProvider,
    _build_local_model_config,
    _hf_pointer_to_https,
)

MODEL_DIR = pathlib.Path(__file__).parent.parent / "models" / "tts" / "pocket_tts" / "english"
VOICE_PATH = pathlib.Path(__file__).parent.parent / "assets" / "pocket_tts_voices" / "sample_voice.safetensors"


def test_hf_pointer_to_https_with_revision():
    url = _hf_pointer_to_https(
        "hf://kyutai/pocket-tts-without-voice-cloning/languages/english/model.safetensors@abc123"
    )
    assert url == (
        "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/"
        "resolve/abc123/languages/english/model.safetensors"
    )


def test_hf_pointer_to_https_without_revision_defaults_to_main():
    url = _hf_pointer_to_https("hf://kyutai/tts-voices/alba-mackenna/casual.wav")
    assert url == "https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/casual.wav"


def test_build_local_model_config_raises_when_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="english"):
        _build_local_model_config("english", tmp_path)


def test_build_local_model_config_prefers_voice_cloning_weights_when_both_present(tmp_path):
    (tmp_path / "tokenizer.model").write_bytes(b"fake")
    (tmp_path / "model.safetensors").write_bytes(b"fake")
    (tmp_path / "model_without_voice_cloning.safetensors").write_bytes(b"fake")

    config_path, used_voice_cloning_weights = _build_local_model_config("english", tmp_path)
    try:
        assert used_voice_cloning_weights is True
        content = config_path.read_text()
        assert str(tmp_path / "model.safetensors") in content
    finally:
        config_path.unlink()


def test_build_local_model_config_falls_back_to_non_cloning_weights(tmp_path):
    (tmp_path / "tokenizer.model").write_bytes(b"fake")
    (tmp_path / "model_without_voice_cloning.safetensors").write_bytes(b"fake")

    config_path, used_voice_cloning_weights = _build_local_model_config("english", tmp_path)
    try:
        assert used_voice_cloning_weights is False
        content = config_path.read_text()
        assert str(tmp_path / "model_without_voice_cloning.safetensors") in content
    finally:
        config_path.unlink()


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def provider():
    assert (MODEL_DIR / "tokenizer.model").exists(), f"missing {MODEL_DIR / 'tokenizer.model'}"
    assert VOICE_PATH.exists(), f"missing {VOICE_PATH}"
    provider = PocketTTSProvider(
        voices={"sample": str(VOICE_PATH)},
        default_voice="sample",
        model_dir=MODEL_DIR,
        language="english",
    )
    yield provider
    provider.unload()


async def _text_stream(*chunks: str):
    for chunk in chunks:
        yield chunk


async def test_synthesize_default_voice_yields_nonempty_audio(provider):
    audio_chunks = [chunk async for chunk in provider.synthesize(_text_stream("Hello there."))]
    assert len(audio_chunks) > 0
    assert sum(len(c) for c in audio_chunks) > 0
    assert all(len(c) % 2 == 0 for c in audio_chunks)  # valid pcm16


async def test_synthesize_handles_multiple_text_chunks_in_order(provider):
    audio_chunks = [
        chunk async for chunk in provider.synthesize(_text_stream("First sentence.", "Second sentence."))
    ]
    assert sum(len(c) for c in audio_chunks) > 0


def test_list_voices_reports_registered_voice(provider):
    voice_ids = {v.id for v in provider.list_voices()}
    assert voice_ids == {"sample"}


def test_output_sample_rate_is_the_wire_rate(provider):
    assert provider.output_sample_rate == 24000
    assert provider.output_format == "pcm16"


async def test_abandoning_synthesis_releases_the_model_lock_promptly(provider):
    first_gen = provider.synthesize(_text_stream("First sentence.", "Second sentence."))
    await first_gen.__anext__()
    await first_gen.aclose()

    start = time.perf_counter()
    second_chunks = [chunk async for chunk in provider.synthesize(_text_stream("Quick."))]
    elapsed = time.perf_counter() - start

    assert sum(len(c) for c in second_chunks) > 0
    assert elapsed < 5.0, f"second synthesize() took {elapsed:.2f}s -- lock from first call may not have released"
