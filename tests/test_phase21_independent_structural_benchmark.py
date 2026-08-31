from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "eval" / "phase21_independent_structural_benchmark.py"
    spec = importlib.util.spec_from_file_location("phase21_independent", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    path = ROOT / "data" / "processed" / "benchmark_fixtures" / "phase21_independent_structural_cases.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_independent_structural_gate_exceeds_single_edge_and_abstains():
    report = _module().build_report(_rows())
    assert report["passed"] is True
    assert report["metrics"]["supported_composition_accuracy"] == 1.0
    assert report["metrics"]["single_edge_supported_accuracy"] == 0.0
    assert report["metrics"]["unsupported_abstention_accuracy"] == 1.0
    assert report["metrics"]["analogy_decision_accuracy"] == 1.0


def test_independent_structural_gate_is_provenance_bound_and_non_promoting():
    report = _module().build_report(_rows())
    assert report["checks"]["external_provenance_bound"] is True
    assert report["checks"]["legacy_fixture_entity_disjoint"] is True
    assert report["checks"]["bounded_observed_only"] is True
    assert report["promotion_ready"] is False
    assert "benchmark-authored" in report["claim_boundary"]


def test_tampered_evidence_hash_fails_closed():
    rows = _rows()
    rows[0]["evidence"]["source_hashes"][0] = "0" * 64
    report = _module().build_report(rows)
    assert report["passed"] is False
    assert report["checks"]["external_provenance_bound"] is False
