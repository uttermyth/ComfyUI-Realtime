"""Node-level smoke tests. Uses the same from-scratch tiny model fixture as
test_transformers_llm_provider.py (see tests/_tiny_transformers_fixture.py)
-- no real checkpoint download needed, so this file is NOT marked
pytest.mark.integration (unlike tests/test_provider_nodes.py, which shares
one module-level integration marker across every other provider node's
tests because those providers do need real downloaded model files)."""
import pathlib

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from comfyui_realtime.nodes.provider_nodes.transformers_llm import (
    TransformersLLMProviderNode,
    _list_transformer_model_dirs,
)
from tests._tiny_transformers_fixture import build_tiny_transformers_model_dir


@pytest.fixture
def models_base_dir(tmp_path) -> pathlib.Path:
    """A fake models/llm/transformers/ base directory containing one valid
    model dir ('tiny-model'), one directory with no config.json (should be
    filtered out), and one stray file (should be filtered out)."""
    base_dir = tmp_path / "llm" / "transformers"
    build_tiny_transformers_model_dir(base_dir / "tiny-model")
    (base_dir / "not-a-model").mkdir(parents=True)
    (base_dir / "not-a-model" / "readme.txt").write_text("no config.json here")
    (base_dir / "stray-file.txt").write_text("not a directory")
    return base_dir


def test_list_transformer_model_dirs_filters_by_config_json(models_base_dir, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda _key: [str(models_base_dir)])
    assert _list_transformer_model_dirs() == ["tiny-model"]


def test_node_execute_returns_a_provider_with_generate(models_base_dir, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda _key: [str(models_base_dir)])
    output = TransformersLLMProviderNode.execute(
        model_name="tiny-model",
        device="cpu",
        torch_dtype="auto",
        trust_remote_code=False,
        system_prompt="",
    )
    (provider,) = output.result
    assert hasattr(provider, "generate")
    provider.unload()
