"""Unit tests for piper voice scanner/resolver helpers in provider_nodes/piper_tts.py.

These tests do not require piper, ComfyUI, or an integration environment.
`folder_paths` is mocked via monkeypatch throughout.
"""
import os
from unittest.mock import patch

import pytest

from comfyui_realtime.nodes.provider_nodes.piper_tts import (
    _list_piper_config_files,
    _list_piper_onnx_files,
    _resolve_piper_voice_path,
)


@pytest.fixture
def voice_dir(tmp_path):
    medium = tmp_path / "en_US" / "lessac" / "medium"
    medium.mkdir(parents=True)
    (medium / "en_US-lessac-medium.onnx").write_bytes(b"fake")
    (medium / "en_US-lessac-medium.onnx.json").write_text("{}")
    return tmp_path


def test_list_onnx_files_returns_only_onnx(voice_dir):
    with patch("folder_paths.get_folder_paths", return_value=[str(voice_dir)]):
        result = _list_piper_onnx_files()
    expected = os.path.join("en_US", "lessac", "medium", "en_US-lessac-medium.onnx")
    assert result == [expected]


def test_list_onnx_files_excludes_json(voice_dir):
    with patch("folder_paths.get_folder_paths", return_value=[str(voice_dir)]):
        result = _list_piper_onnx_files()
    assert not any(f.endswith(".json") for f in result)


def test_list_config_files_returns_only_onnx_json(voice_dir):
    with patch("folder_paths.get_folder_paths", return_value=[str(voice_dir)]):
        result = _list_piper_config_files()
    expected = os.path.join("en_US", "lessac", "medium", "en_US-lessac-medium.onnx.json")
    assert result == [expected]


def test_list_config_files_excludes_plain_onnx(voice_dir):
    with patch("folder_paths.get_folder_paths", return_value=[str(voice_dir)]):
        result = _list_piper_config_files()
    assert not any(f.endswith(".onnx") and not f.endswith(".onnx.json") for f in result)


def test_list_onnx_files_empty_when_dir_missing():
    with patch("folder_paths.get_folder_paths", return_value=["/nonexistent/path"]):
        result = _list_piper_onnx_files()
    assert result == []


def test_list_onnx_files_sorted(tmp_path):
    for name in ["beta.onnx", "alpha.onnx"]:
        (tmp_path / name).write_bytes(b"")
    with patch("folder_paths.get_folder_paths", return_value=[str(tmp_path)]):
        result = _list_piper_onnx_files()
    assert result == sorted(result)


def test_resolve_annotated_input_path():
    with patch("folder_paths.get_annotated_filepath", return_value="/comfy/input/voice.onnx") as mock_get:
        result = _resolve_piper_voice_path("voice.onnx [input]")
    mock_get.assert_called_once_with("voice.onnx [input]")
    assert result == "/comfy/input/voice.onnx"


def test_resolve_annotated_output_path():
    with patch("folder_paths.get_annotated_filepath", return_value="/comfy/output/voice.onnx") as mock_get:
        result = _resolve_piper_voice_path("voice.onnx [output]")
    mock_get.assert_called_once_with("voice.onnx [output]")
    assert result == "/comfy/output/voice.onnx"


def test_resolve_annotated_temp_path():
    with patch("folder_paths.get_annotated_filepath", return_value="/comfy/temp/voice.onnx") as mock_get:
        result = _resolve_piper_voice_path("voice.onnx [temp]")
    mock_get.assert_called_once_with("voice.onnx [temp]")
    assert result == "/comfy/temp/voice.onnx"


def test_resolve_absolute_path_returned_as_is():
    result = _resolve_piper_voice_path("/absolute/path/to/voice.onnx")
    assert result == "/absolute/path/to/voice.onnx"


def test_resolve_relative_path_against_registered_dir(voice_dir):
    rel = os.path.join("en_US", "lessac", "medium", "en_US-lessac-medium.onnx")
    with patch("folder_paths.get_folder_paths", return_value=[str(voice_dir)]):
        result = _resolve_piper_voice_path(rel)
    expected = str(voice_dir / "en_US" / "lessac" / "medium" / "en_US-lessac-medium.onnx")
    assert result == expected


def test_resolve_unmatched_relative_path_returned_unchanged():
    with patch("folder_paths.get_folder_paths", return_value=["/nonexistent"]):
        result = _resolve_piper_voice_path("some/missing/voice.onnx")
    assert result == "some/missing/voice.onnx"
