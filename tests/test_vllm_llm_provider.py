"""VLLMProvider tests run entirely against a stub AsyncLLMEngine (see
tests/_stub_vllm_engine.py) -- vLLM's real engine requires a CUDA GPU this
test suite cannot assume is present. Real engine/model behavior is manual
verification only, per the design spec's Testing Strategy section."""
import pytest

pytest.importorskip("vllm")

from comfyui_realtime.providers import vllm_llm
from comfyui_realtime.providers.base import ChatMessage, GenerationOptions
from comfyui_realtime.providers.vllm_llm import VLLMProvider
from tests._stub_vllm_engine import (
    NoChatTemplateTokenizer,
    StubTokenizer,
    make_stub_engine_class,
)


def _patch_engine_and_tokenizer(monkeypatch, *, engine_class=None, tokenizer=None):
    monkeypatch.setattr(vllm_llm, "AsyncLLMEngine", engine_class or make_stub_engine_class())
    monkeypatch.setattr(vllm_llm, "get_tokenizer", lambda *a, **k: tokenizer or StubTokenizer())


def test_construction_raises_if_no_cuda_gpu(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        VLLMProvider(model_path="/fake/path")


def test_construction_raises_if_tokenizer_has_no_chat_template(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    _patch_engine_and_tokenizer(monkeypatch, tokenizer=NoChatTemplateTokenizer())
    with pytest.raises(ValueError, match="chat_template"):
        VLLMProvider(model_path="/fake/path")


def test_construction_builds_engine_args_from_model_path(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class()
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)

    VLLMProvider(model_path="/fake/path", gpu_memory_utilization=0.5)

    (instance,) = StubEngine.instances
    assert instance.engine_args.model == "/fake/path"
    assert instance.engine_args.gpu_memory_utilization == 0.5


def test_construction_leaves_max_model_len_dtype_quantization_unset_by_default(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class()
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)

    VLLMProvider(model_path="/fake/path")

    (instance,) = StubEngine.instances
    default_args = type(instance.engine_args)(model="/fake/path")
    assert instance.engine_args.max_model_len == default_args.max_model_len
    assert instance.engine_args.dtype == default_args.dtype
    assert instance.engine_args.quantization == default_args.quantization


def test_construction_applies_explicit_max_model_len_dtype_quantization(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class()
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)

    VLLMProvider(
        model_path="/fake/path",
        max_model_len=4096,
        dtype="bfloat16",
        quantization="modelopt",
    )

    (instance,) = StubEngine.instances
    assert instance.engine_args.max_model_len == 4096
    assert instance.engine_args.dtype == "bfloat16"
    assert instance.engine_args.quantization == "modelopt"


def test_engine_args_json_overlay_applies_valid_overrides(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    StubEngine = make_stub_engine_class()
    _patch_engine_and_tokenizer(monkeypatch, engine_class=StubEngine)

    VLLMProvider(model_path="/fake/path", engine_args='{"tensor_parallel_size": 2}')

    (instance,) = StubEngine.instances
    assert instance.engine_args.tensor_parallel_size == 2


def test_engine_args_invalid_json_raises_value_error(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    _patch_engine_and_tokenizer(monkeypatch)
    with pytest.raises(ValueError, match="not valid JSON"):
        VLLMProvider(model_path="/fake/path", engine_args="{not json")


def test_engine_args_unknown_key_raises_value_error_with_hint(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    _patch_engine_and_tokenizer(monkeypatch)
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        VLLMProvider(model_path="/fake/path", engine_args='{"tensor_paralel_size": 2}')


def test_unload_clears_engine_and_tokenizer(monkeypatch):
    monkeypatch.setattr(vllm_llm.torch.cuda, "is_available", lambda: True)
    _patch_engine_and_tokenizer(monkeypatch)
    provider = VLLMProvider(model_path="/fake/path")
    provider.unload()
    assert provider._engine is None
    assert provider._tokenizer is None
