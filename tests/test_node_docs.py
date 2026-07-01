import pytest
import folder_paths

from comfyui_realtime.nodes.pipeline_node import RealtimePipelineNode
from comfyui_realtime.nodes.provider_nodes.llama_cpp_llm import LlamaCppLLMProviderNode
from comfyui_realtime.nodes.provider_nodes.faster_whisper_stt import FasterWhisperSTTProviderNode
from comfyui_realtime.nodes.provider_nodes.whisper_cpp_stt import WhisperCppSTTProviderNode
from comfyui_realtime.nodes.provider_nodes.silero_vad import SileroVADProviderNode
from comfyui_realtime.nodes.provider_nodes.piper_tts import PiperTTSProviderNode


@pytest.fixture(autouse=True)
def mock_model_lists(monkeypatch):
    # LlamaCppLLM and WhisperCpp call folder_paths.get_filename_list() inside
    # define_schema() to populate their Combo options — return empty so tests
    # don't depend on files being present on disk.
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda _: [])


@pytest.mark.parametrize("node_cls", [
    RealtimePipelineNode,
    LlamaCppLLMProviderNode,
    FasterWhisperSTTProviderNode,
    WhisperCppSTTProviderNode,
    SileroVADProviderNode,
    PiperTTSProviderNode,
])
def test_schema_has_description(node_cls):
    schema = node_cls.define_schema()
    assert schema.description, f"{node_cls.__name__} schema.description must be non-empty"


@pytest.mark.parametrize("node_cls", [
    RealtimePipelineNode,
    LlamaCppLLMProviderNode,
    FasterWhisperSTTProviderNode,
    WhisperCppSTTProviderNode,
    SileroVADProviderNode,
    PiperTTSProviderNode,
])
def test_all_inputs_have_tooltips(node_cls):
    schema = node_cls.define_schema()
    missing = [inp.id for inp in schema.inputs if not inp.tooltip]
    assert not missing, f"{node_cls.__name__} inputs missing tooltip: {missing}"
