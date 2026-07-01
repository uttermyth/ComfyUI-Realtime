# tests/test_provider_nodes.py
"""Node-level smoke tests -- just confirm define_schema()/execute() wiring is correct."""
import json
import pathlib

import pytest

pytest.importorskip("llama_cpp")
pytest.importorskip("piper")
pytest.importorskip("silero_vad")
pytest.importorskip("pywhispercpp")
pytest.importorskip("pocket_tts")

from comfyui_realtime.nodes.provider_nodes import (
    LlamaCppLLMProviderNode,
    PiperTTSProviderNode,
    PocketTTSProviderNode,
    SileroVADProviderNode,
    WhisperCppSTTProviderNode,
)

MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"
VOICES_DIR = pathlib.Path(__file__).parent.parent / "assets" / "piper_voices" / "en_US" / "lessac"
ONNX_PATH = str(VOICES_DIR / "medium" / "en_US-lessac-medium.onnx")
STT_MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "ggml-base.en.bin"
POCKET_TTS_MODELS_DIR = pathlib.Path(__file__).parent.parent / "models"
POCKET_TTS_VOICE_PATH = str(
    pathlib.Path(__file__).parent.parent / "assets" / "pocket_tts_voices" / "sample_voice.safetensors"
)

pytestmark = pytest.mark.integration


def test_llama_cpp_node_returns_a_provider_with_generate(monkeypatch):
    monkeypatch.setattr(
        "folder_paths.get_full_path_or_raise",
        lambda _folder, _name: str(MODEL_PATH),
    )
    output = LlamaCppLLMProviderNode.execute(
        model_name=MODEL_PATH.name, n_ctx=512, n_gpu_layers=0, system_prompt=""
    )
    (provider,) = output.result
    assert hasattr(provider, "generate")
    provider.unload()


def test_piper_node_returns_a_provider_with_synthesize_and_default_voice():
    output = PiperTTSProviderNode.execute(
        default_voice_path=ONNX_PATH,
        default_voice_id="lessac-medium",
    )
    (provider,) = output.result
    assert hasattr(provider, "synthesize")
    voice_ids = {v.id for v in provider.list_voices()}
    assert "lessac-medium" in voice_ids
    provider.unload()


def test_piper_node_auto_derives_config_path():
    output = PiperTTSProviderNode.execute(
        default_voice_path=ONNX_PATH,
        default_voice_id="lessac-medium",
    )
    (provider,) = output.result
    # Provider loaded successfully, proving config was found at onnx_path + ".json"
    assert hasattr(provider, "synthesize")
    provider.unload()


def test_piper_node_with_no_additional_voices_behaves_exactly_as_before():
    output = PiperTTSProviderNode.execute(
        default_voice_path=ONNX_PATH,
        default_voice_id="lessac-medium",
    )
    (provider,) = output.result
    voice_ids = {v.id for v in provider.list_voices()}
    assert voice_ids == {"lessac-medium"}
    provider.unload()


def test_piper_node_merges_additional_voices_into_one_provider():
    additional = json.dumps(
        [
            {
                "id": "lessac-medium-2",
                "onnx_path": ONNX_PATH,
                "config_path": ONNX_PATH + ".json",
            }
        ]
    )
    output = PiperTTSProviderNode.execute(
        default_voice_path=ONNX_PATH,
        default_voice_id="lessac-medium",
        additional_voices_json=additional,
    )
    (provider,) = output.result
    voice_ids = {v.id for v in provider.list_voices()}
    assert voice_ids == {"lessac-medium", "lessac-medium-2"}
    provider.unload()


def test_piper_node_rejects_malformed_additional_voices_json():
    with pytest.raises(ValueError, match="additional_voices_json"):
        PiperTTSProviderNode.execute(
            default_voice_path=ONNX_PATH,
            default_voice_id="lessac-medium",
            additional_voices_json="not valid json",
        )


def test_piper_node_rejects_additional_voice_entry_missing_a_required_key():
    bad_entry = json.dumps([{"id": "x", "onnx_path": "/tmp/x.onnx"}])  # missing config_path
    with pytest.raises(ValueError, match="config_path"):
        PiperTTSProviderNode.execute(
            default_voice_path=ONNX_PATH,
            default_voice_id="lessac-medium",
            additional_voices_json=bad_entry,
        )


def test_silero_vad_node_returns_a_provider_with_analyze():
    output = SileroVADProviderNode.execute(threshold=0.5)
    (provider,) = output.result
    assert hasattr(provider, "analyze")
    provider.unload()


def test_whisper_cpp_stt_node_returns_a_provider_with_transcribe(monkeypatch):
    monkeypatch.setattr(
        "folder_paths.get_full_path_or_raise",
        lambda _folder, _name: str(STT_MODEL_PATH),
    )
    output = WhisperCppSTTProviderNode.execute(model_name=STT_MODEL_PATH.name)
    (provider,) = output.result
    assert hasattr(provider, "transcribe")
    provider.unload()


def test_pocket_tts_node_returns_a_provider_with_synthesize_and_default_voice(monkeypatch):
    monkeypatch.setattr("folder_paths.models_dir", str(POCKET_TTS_MODELS_DIR))
    output = PocketTTSProviderNode.execute(
        language="english",
        default_voice_path=POCKET_TTS_VOICE_PATH,
        default_voice_id="sample",
    )
    (provider,) = output.result
    assert hasattr(provider, "synthesize")
    voice_ids = {v.id for v in provider.list_voices()}
    assert voice_ids == {"sample"}
    provider.unload()


def test_pocket_tts_node_merges_additional_voices_into_one_provider(monkeypatch):
    monkeypatch.setattr("folder_paths.models_dir", str(POCKET_TTS_MODELS_DIR))
    additional = json.dumps([{"id": "sample-2", "voice_source": POCKET_TTS_VOICE_PATH}])
    output = PocketTTSProviderNode.execute(
        language="english",
        default_voice_path=POCKET_TTS_VOICE_PATH,
        default_voice_id="sample",
        additional_voices_json=additional,
    )
    (provider,) = output.result
    voice_ids = {v.id for v in provider.list_voices()}
    assert voice_ids == {"sample", "sample-2"}
    provider.unload()


def test_pocket_tts_node_rejects_malformed_additional_voices_json():
    with pytest.raises(ValueError, match="additional_voices_json"):
        PocketTTSProviderNode.execute(
            language="english",
            default_voice_path=POCKET_TTS_VOICE_PATH,
            default_voice_id="sample",
            additional_voices_json="not valid json",
        )


def test_pocket_tts_node_rejects_additional_voice_entry_missing_a_required_key():
    bad_entry = json.dumps([{"id": "x"}])  # missing voice_source
    with pytest.raises(ValueError, match="voice_source"):
        PocketTTSProviderNode.execute(
            language="english",
            default_voice_path=POCKET_TTS_VOICE_PATH,
            default_voice_id="sample",
            additional_voices_json=bad_entry,
        )
