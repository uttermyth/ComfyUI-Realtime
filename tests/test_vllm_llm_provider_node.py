"""Node-level smoke tests, using the same stub-engine approach as
tests/test_vllm_llm_provider.py -- see that file's module docstring for why
(no real GPU/vLLM engine is exercised)."""
import pathlib

import pytest

pytest.importorskip("vllm")

from comfyui_realtime.nodes.provider_nodes import vllm_llm as vllm_llm_node
from comfyui_realtime.nodes.provider_nodes.vllm_llm import VLLMProviderNode
from comfyui_realtime.providers import vllm_llm as vllm_llm_provider
from tests._stub_vllm_engine import StubTokenizer, make_stub_engine_class


@pytest.fixture
def models_base_dir(tmp_path) -> pathlib.Path:
    base_dir = tmp_path / "llm" / "transformers"
    model_dir = base_dir / "tiny-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    return base_dir


def _patch_engine_and_tokenizer(monkeypatch):
    monkeypatch.setattr(vllm_llm_provider.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vllm_llm_provider, "AsyncLLMEngine", make_stub_engine_class())
    monkeypatch.setattr(vllm_llm_provider, "get_tokenizer", lambda *a, **k: StubTokenizer())


def test_node_execute_returns_a_provider_with_generate(models_base_dir, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda _key: [str(models_base_dir)])
    _patch_engine_and_tokenizer(monkeypatch)

    output = VLLMProviderNode.execute(
        model_name="tiny-model",
        gpu_memory_utilization=0.9,
        max_model_len=0,
        dtype="auto",
        quantization="",
        enforce_eager=False,
        trust_remote_code=False,
        engine_args="",
        system_prompt="",
    )
    (provider,) = output.result
    assert hasattr(provider, "generate")
    provider.unload()


def test_node_execute_raises_file_not_found_for_missing_model_dir(models_base_dir, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda _key: [str(models_base_dir)])
    _patch_engine_and_tokenizer(monkeypatch)

    with pytest.raises(FileNotFoundError):
        VLLMProviderNode.execute(
            model_name="does-not-exist",
            gpu_memory_utilization=0.9,
            max_model_len=0,
            dtype="auto",
            quantization="",
            enforce_eager=False,
            trust_remote_code=False,
            engine_args="",
            system_prompt="",
        )


def test_node_execute_converts_max_model_len_zero_to_none(models_base_dir, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda _key: [str(models_base_dir)])
    StubEngine = make_stub_engine_class()
    monkeypatch.setattr(vllm_llm_provider.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vllm_llm_provider, "AsyncLLMEngine", StubEngine)
    monkeypatch.setattr(vllm_llm_provider, "get_tokenizer", lambda *a, **k: StubTokenizer())

    VLLMProviderNode.execute(
        model_name="tiny-model",
        gpu_memory_utilization=0.9,
        max_model_len=0,
        dtype="auto",
        quantization="",
        enforce_eager=False,
        trust_remote_code=False,
        engine_args="",
        system_prompt="",
    )

    (instance,) = StubEngine.instances
    default_args = type(instance.engine_args)(model=instance.engine_args.model)
    assert instance.engine_args.max_model_len == default_args.max_model_len
