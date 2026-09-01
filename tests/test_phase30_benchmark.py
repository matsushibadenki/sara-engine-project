from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.phase30_benchmark import evaluate_frozen_decisions, freeze_decisions
from sara_engine.evaluation.phase30_fixtures import build_fixtures


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = json.loads((ROOT / "workspace/evaluation/phase30_temporal_effective_interaction_preregistration.json").read_text(encoding="utf-8"))


def _bundle():
    inputs, keys, manifest = build_fixtures()
    decisions, identity = freeze_decisions(inputs)
    return inputs, keys, manifest, decisions, identity


def test_phase30_decisions_freeze_before_evaluator_access():
    inputs, _, manifest, decisions, identity = _bundle()
    assert identity["evaluator_key_loaded"] is False
    assert identity["input_digest"] == manifest["input_digest"]
    assert identity["decision_count"] == len(inputs) * 4 == 520
    assert all("expected_decision" not in row for row in decisions)


def test_phase30_benchmark_is_complete_fail_closed_and_non_promoting():
    inputs, keys, manifest, decisions, identity = _bundle()
    report = evaluate_frozen_decisions(inputs, keys, manifest, PREREGISTRATION, decisions, identity)
    assert report["decision_count"] == 520
    assert report["evaluation_decision_count"] == 260
    assert set(report["arm_metrics"]) == set(PREREGISTRATION["arms"])
    assert set(report["threshold_results"]) == set(PREREGISTRATION["thresholds"])
    assert report["promotion_ready"] is False
    assert report["production_mutation"] is False
    assert len(report["report_digest"]) == 64


@pytest.mark.parametrize("target", ("decision", "identity", "key"))
def test_phase30_benchmark_rejects_tampering(target):
    inputs, keys, manifest, decisions, identity = _bundle()
    if target == "decision":
        decisions[0]["decision"] = "tampered"
    elif target == "identity":
        identity["evaluator_key_loaded"] = True
    else:
        keys[0]["expected_decision"] = "tampered"
    with pytest.raises(ValueError):
        evaluate_frozen_decisions(inputs, keys, manifest, PREREGISTRATION, decisions, identity)
