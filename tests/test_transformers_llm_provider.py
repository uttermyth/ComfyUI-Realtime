"""No real checkpoint download needed -- see tests/_tiny_transformers_fixture.py.
Not marked pytest.mark.integration (unlike test_llama_cpp_llm_provider.py /
test_pocket_tts_provider.py) since this provider's fixture model is built
from scratch locally and runs fast even on CPU."""
import pathlib

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

import torch

from comfyui_realtime.providers.transformers_llm import TransformersLLMProvider
from tests._tiny_transformers_fixture import build_tiny_transformers_model_dir


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> pathlib.Path:
    target_dir = tmp_path_factory.mktemp("tiny_transformers_model")
    build_tiny_transformers_model_dir(target_dir)
    return target_dir


@pytest.fixture(scope="module")
def tiny_model_dir_without_chat_template(tmp_path_factory) -> pathlib.Path:
    target_dir = tmp_path_factory.mktemp("tiny_transformers_model_no_template")
    build_tiny_transformers_model_dir(target_dir, include_chat_template=False)
    return target_dir


def test_construction_succeeds_on_cpu(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        assert hasattr(provider, "generate")
    finally:
        provider.unload()


def test_construction_raises_when_chat_template_missing(tiny_model_dir_without_chat_template):
    with pytest.raises(ValueError, match="chat_template"):
        TransformersLLMProvider(model_path=str(tiny_model_dir_without_chat_template), device="cpu")


def test_construction_raises_for_unavailable_cuda_device(tiny_model_dir, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        TransformersLLMProvider(model_path=str(tiny_model_dir), device="cuda")


def test_construction_raises_for_unavailable_mps_device(tiny_model_dir, monkeypatch):
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="mps"):
        TransformersLLMProvider(model_path=str(tiny_model_dir), device="mps")


def test_auto_dtype_resolves_to_float32_on_cpu(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu", torch_dtype="auto")
    try:
        assert next(provider._model.parameters()).dtype == torch.float32
    finally:
        provider.unload()


def test_explicit_dtype_is_respected(tiny_model_dir):
    provider = TransformersLLMProvider(
        model_path=str(tiny_model_dir), device="cpu", torch_dtype="bfloat16"
    )
    try:
        assert next(provider._model.parameters()).dtype == torch.bfloat16
    finally:
        provider.unload()


def test_trust_remote_code_defaults_to_false(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        assert provider._trust_remote_code is False
    finally:
        provider.unload()


def test_unload_releases_the_model(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    provider.unload()
    assert provider._model is None
