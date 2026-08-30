from __future__ import annotations

import json

import pytest

from sara_engine.edge.portable_decision_trace import (
    canonical_decision_json,
    canonicalize_decisions,
    decision_trace_digest,
)


def _record(decision_id: str, subsystem: str, **overrides):
    record = {
        "decision_id": decision_id,
        "sequence": 0,
        "subsystem": subsystem,
        "subject_id": "subject-日本",
        "evidence_ids": ["evidence-b", "evidence-a", "evidence-a"],
        "verified": True,
        "contradiction": False,
        "stale": False,
        "capacity_available": True,
        "prediction_match": True,
        "support_count": 1,
    }
    record.update(overrides)
    return record


def test_portable_decision_rules_cover_three_subsystem_boundaries():
    records = [
        _record("memory-admit", "event_memory"),
        _record("memory-conflict", "event_memory", contradiction=True),
        _record("risa-propose", "risa_proposal"),
        _record("risa-missing", "risa_proposal", support_count=0),
        _record("feedback-retain", "predictive_feedback"),
        _record("feedback-correct", "predictive_feedback", prediction_match=False),
    ]
    decisions = {
        row["decision_id"]: row["decision"] for row in canonicalize_decisions(records)
    }
    assert decisions == {
        "memory-admit": "admit",
        "memory-conflict": "reject_contradiction",
        "risa-propose": "propose",
        "risa-missing": "reject_missing_support",
        "feedback-retain": "retain_prediction",
        "feedback-correct": "emit_correction",
    }
    assert "\\u65e5\\u672c" in canonical_decision_json(records)
    assert len(decision_trace_digest(records)) == 64


def test_rust_portable_decision_trace_matches_python_bytes_and_digest():
    rust = pytest.importorskip("sara_engine.sara_rust_core")
    records = [
        _record("feedback-correct", "predictive_feedback", sequence=2, prediction_match=False),
        _record("memory-admit", "event_memory", sequence=0),
        _record("risa-conflict", "risa_proposal", sequence=1, contradiction=True),
    ]
    source = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    assert rust.canonical_portable_decision_trace_json(source) == canonical_decision_json(records)
    assert rust.portable_decision_trace_digest(source) == decision_trace_digest(records)


def test_portable_decision_trace_rejects_duplicate_and_unknown_subsystem():
    row = _record("duplicate", "event_memory")
    with pytest.raises(ValueError, match="duplicate decision_id"):
        canonicalize_decisions([row, row])
    with pytest.raises(ValueError, match="unsupported subsystem"):
        canonicalize_decisions([_record("unknown", "other")])
