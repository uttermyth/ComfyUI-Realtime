"""Shared HuggingFace transformers-format model directory discovery, used
by both TransformersLLMProviderNode and VLLMProviderNode -- both consume
the same on-disk model pool (models/llm/transformers/, folder_paths key
"llm-transformers") to avoid duplicating
large checkpoint files on disk for two providers that read the identical
directory format (config.json + .safetensors shards + tokenizer files).
"""
from __future__ import annotations

import os

import folder_paths


def _list_transformer_model_dirs() -> list[str]:
    """List immediate subdirectories of every registered llm-transformers
    base path that contain a config.json."""
    names = []
    for base_dir in folder_paths.get_folder_paths("llm-transformers"):
        if not os.path.isdir(base_dir):
            continue
        for entry in sorted(os.listdir(base_dir)):
            entry_path = os.path.join(base_dir, entry)
            if os.path.isdir(entry_path) and os.path.isfile(os.path.join(entry_path, "config.json")):
                names.append(entry)
    return names
