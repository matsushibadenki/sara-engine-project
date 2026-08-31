from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "eval" / "phase37_source_split_review.py"
    spec = importlib.util.spec_from_file_location("phase37_source_review", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _draft():
    return json.loads((ROOT / "workspace" / "evaluation" / "phase37_source_split_review_draft.json").read_text(encoding="utf-8"))


def test_phase37_source_packet_passes_automated_checks_but_requires_human_review():
    packet = _module().build_packet(_draft())
    assert packet["automated_integrity_passed"] is True
    assert packet["human_review_complete"] is False
    assert packet["fixture_freeze_allowed"] is False
    assert packet["source_count"] == 8
    assert packet["train_count"] == packet["evaluation_count"] == 4


def test_phase37_all_explicit_human_approvals_open_freeze_gate():
    draft = _draft()
    decisions = {"decisions": [{"record_id": row["record_id"], "decision": "approve", "reviewer": "human_operator"} for row in draft["sources"]]}
    packet = _module().build_packet(draft, decisions)
    assert packet["human_review_complete"] is True
    assert packet["fixture_freeze_allowed"] is True


def test_phase37_hash_tampering_fails_automated_gate():
    draft = _draft()
    draft["sources"][0]["source_hash"] = "0" * 64
    packet = _module().build_packet(draft)
    assert packet["automated_integrity_passed"] is False
    assert packet["fixture_freeze_allowed"] is False
