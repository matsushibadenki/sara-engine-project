from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "eval" / "phase37_freeze_source_split.py"
    spec = importlib.util.spec_from_file_location("phase37_freeze", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs():
    draft = json.loads((ROOT / "workspace" / "evaluation" / "phase37_source_split_review_draft.json").read_text(encoding="utf-8"))
    decisions = json.loads((ROOT / "workspace" / "evaluation" / "phase37_source_split_human_review_decisions.json").read_text(encoding="utf-8"))
    return draft, decisions


def test_approved_phase37_sources_build_frozen_base_artifacts():
    artifacts = _module().build_artifacts(*_inputs())
    assert artifacts["packet"]["fixture_freeze_allowed"] is True
    assert all(len(value) == 64 for value in artifacts["hashes"].values())
    train = [json.loads(line) for line in artifacts["payloads"]["train_fixture"].splitlines()]
    evaluation = [json.loads(line) for line in artifacts["payloads"]["evaluation_fixture"].splitlines()]
    assert len(train) == len(evaluation) == 4
    assert all(row["withheld_edge"] is None for row in train)
    assert all(row["withheld_edge"] is not None for row in evaluation)
    assert {row["structural_family"] for row in train}.isdisjoint({row["structural_family"] for row in evaluation})


def test_phase37_decisions_must_bind_exact_draft():
    draft, decisions = _inputs()
    decisions = copy.deepcopy(decisions)
    decisions["draft_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="do not bind"):
        _module().build_artifacts(draft, decisions)


def test_phase37_missing_approval_keeps_freeze_closed():
    draft, decisions = _inputs()
    decisions = copy.deepcopy(decisions)
    decisions["decisions"].pop()
    with pytest.raises(ValueError, match="gate is closed"):
        _module().build_artifacts(draft, decisions)
