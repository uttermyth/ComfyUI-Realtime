"""Structural validation only -- these workflows use portable placeholder
paths (spec section 10, Phase 4) that don't resolve to real files on any
machine, so they are never executed here, only checked for well-formed
JSON and that every referenced node class actually exists in the real,
assembled node set. Same access pattern as tests/test_node_registration.py
for reading it under pytest (comfy-org v3 migration: via
discover_v3_node_classes(), not a static NODE_CLASS_MAPPINGS dict)."""
from __future__ import annotations

import json
import pathlib

import pytest

from .conftest import discover_v3_node_classes

WORKFLOWS_DIR = pathlib.Path(__file__).parent.parent / "workflows" / "api"


def _workflow_files():
    return sorted(WORKFLOWS_DIR.glob("*.json"))


def test_at_least_six_example_workflows_exist():
    assert len(_workflow_files()) >= 6, (
        f"expected at least 6 example workflows (one per pipeline shape plus a "
        f"Uttermyth-style example), found {len(_workflow_files())}: {[p.name for p in _workflow_files()]}"
    )


@pytest.mark.parametrize("workflow_path", _workflow_files(), ids=lambda p: p.name)
def test_example_workflow_is_well_formed(workflow_path):
    with open(workflow_path) as f:
        data = json.load(f)
    assert "prompt" in data
    assert isinstance(data["prompt"], dict)
    assert len(data["prompt"]) > 0
    assert "client_id" in data


@pytest.mark.parametrize("workflow_path", _workflow_files(), ids=lambda p: p.name)
def test_example_workflow_references_only_real_node_classes(workflow_path):
    registered_classes = discover_v3_node_classes()
    with open(workflow_path) as f:
        data = json.load(f)
    referenced = {node["class_type"] for node in data["prompt"].values()}
    unknown = referenced - set(registered_classes.keys())
    assert not unknown, f"{workflow_path.name} references unknown node class(es): {unknown}"


@pytest.mark.parametrize("workflow_path", _workflow_files(), ids=lambda p: p.name)
def test_example_workflow_includes_a_realtime_pipeline_node(workflow_path):
    with open(workflow_path) as f:
        data = json.load(f)
    class_types = {node["class_type"] for node in data["prompt"].values()}
    assert "RealtimePipelineNode" in class_types, f"{workflow_path.name} never registers a pipeline"
