from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "eval" / "phase37_structural_invariant_benchmark.py"
    spec = importlib.util.spec_from_file_location("phase37_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase37_first_attempt_is_retained_without_promotion():
    module = _module()
    with open(module.REGISTRATION, encoding="utf-8") as handle:
        registration = json.load(handle)
    report = module.build_report(module._rows(module.TRAIN), module._rows(module.INPUTS), module._rows(module.KEY), registration)
    assert report["single_registered_attempt_consumed"] is True
    assert report["passed"] is False
    assert report["retained_negative_result"] is True
    assert report["promotion_ready"] is False
    assert report["checks"]["all_proposals_provisional"] is True
    assert report["checks"]["resource_bounds_passed"] is True


def test_phase37_candidate_trace_never_contains_evaluator_labels():
    module = _module()
    with open(module.REGISTRATION, encoding="utf-8") as handle:
        registration = json.load(handle)
    report = module.build_report(module._rows(module.TRAIN), module._rows(module.INPUTS), module._rows(module.KEY), registration)
    traces = json.dumps([row["candidate"] for row in report["results"]], sort_keys=True)
    assert "expected_decision" not in traces
    assert "withheld_edge" not in traces
    assert "case_family" not in traces
