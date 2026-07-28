from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "next_level_promotion_review.py"
    spec = importlib.util.spec_from_file_location("next_level_promotion_review", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_next_level_review_blocks_promotion_and_surfaces_negative_results(tmp_path):
    module = _load_module()
    reports = {
        "next_level_structural_benchmark.json": {"passed": True},
        "continual_horizon_benchmark.json": {"passed": True},
        "continual_horizon_external_gate.json": {"passed": True, "promotion_allowed": False},
        "phase23_structural_fusion_benchmark.json": {"passed": True},
        "phase24_causal_benchmark.json": {"passed": True},
        "phase25_agent_loop_benchmark.json": {"passed": True},
    }
    for name, payload in reports.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    review = module.build_review(str(tmp_path))

    assert review["promotion_allowed"] is False
    assert review["checks"]["internal_phase_evidence_complete"] is True
    assert review["negative_results"]
    assert review["next_actions"]


def test_next_level_review_marks_missing_phase_evidence():
    module = _load_module()
    review = module.build_review("/nonexistent/evaluation")

    assert review["checks"]["internal_phase_evidence_complete"] is False
    assert review["promotion_allowed"] is False


def test_next_level_review_accepts_current_evidence_bound_human_approval(tmp_path):
    module = _load_module()
    from sara_engine.evaluation.promotion_approval import build_approval

    reports = {
        "next_level_structural_benchmark.json": {"passed": True},
        "continual_horizon_benchmark.json": {"passed": True},
        "continual_horizon_external_gate.json": {"passed": True, "promotion_allowed": True},
        "phase23_structural_fusion_benchmark.json": {
            "passed": True,
        },
        "phase23_external_multimodal_gate.json": {"passed": True, "promotion_allowed": True},
        "phase24_causal_benchmark.json": {"passed": True},
        "phase25_agent_loop_benchmark.json": {"passed": True},
    }
    for name, payload in reports.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    loaded = {
        key: module._read_json(str(tmp_path / filename))
        for key, filename in module.REPORT_FILES.items()
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(build_approval(loaded, reviewer="operator")),
        encoding="utf-8",
    )

    review = module.build_review(str(tmp_path), str(approval_path))

    assert review["checks"]["human_approval_valid"] is True
    assert review["promotion_allowed"] is True


def test_next_level_review_suppresses_repeated_failed_internal_rerun(
    tmp_path, monkeypatch
):
    module = _load_module()
    monkeypatch.setattr(module, "ensure_parent_directory", lambda path: path)
    reports = {
        "next_level_structural_benchmark.json": {"passed": True},
        "continual_horizon_benchmark.json": {"passed": True},
        "continual_horizon_external_gate.json": {
            "passed": True,
            "promotion_allowed": False,
        },
        "phase23_structural_fusion_benchmark.json": {"passed": True},
        "phase23_external_multimodal_gate.json": {
            "passed": False,
            "promotion_allowed": False,
        },
        "phase24_causal_benchmark.json": {"passed": True},
        "phase25_agent_loop_benchmark.json": {"passed": True},
    }
    for name, payload in reports.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    journal = tmp_path / "journal.jsonl"
    report = tmp_path / "review.json"
    gate = tmp_path / "gate.json"

    first = module.build_review(str(tmp_path), journal_path=str(journal))
    module.write_outputs(first, str(report), str(gate), str(journal))
    second = module.build_review(str(tmp_path), journal_path=str(journal))
    module.write_outputs(second, str(report), str(gate), str(journal))

    assert first["checks"]["repeated_failed_experiment_detected"] is False
    assert second["checks"]["repeated_failed_experiment_detected"] is True
    assert second["checks"]["duplicate_work_suppressed"] is True
    assert second["research_memory"]["prior_match_count"] == 1
    assert [item["action"] for item in second["next_actions"]] == [
        "collect_independent_multimodal_records",
        "collect_independent_horizon_records",
    ]
    assert second["research_memory"]["suppressed_actions"][0]["action"] == (
        "rerun_internal_phase_benchmarks"
    )
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 1


def test_next_level_review_does_not_suppress_changed_failed_experiment(tmp_path):
    module = _load_module()
    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        json.dumps(
            {
                "schema": "sara-next-level-research-journal-v1",
                "hypothesis": "different hypothesis",
                "evidence": {},
                "negative_results": ["different failure"],
                "next_tests": [],
                "promotion_allowed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    review = module.build_review(str(tmp_path), journal_path=str(journal))

    assert review["research_memory"]["prior_match_count"] == 0
    assert review["checks"]["duplicate_work_suppressed"] is False
    assert any(
        item["action"] == "rerun_internal_phase_benchmarks"
        for item in review["next_actions"]
    )


def test_next_level_review_blocks_unexplained_metric_regression(tmp_path):
    module = _load_module()
    reports = {
        "next_level_structural_benchmark.json": {
            "passed": True,
            "metrics": {"decision_accuracy": 1.0},
            "cases": {"case-a": {"source_hash": "stable-source"}},
        },
        "continual_horizon_benchmark.json": {"passed": True, "metrics": {}},
        "continual_horizon_external_gate.json": {
            "passed": True,
            "promotion_allowed": False,
            "metrics": {},
        },
        "phase23_structural_fusion_benchmark.json": {"passed": True, "metrics": {}},
        "phase23_external_multimodal_gate.json": {
            "passed": False,
            "promotion_allowed": False,
            "metrics": {},
        },
        "phase24_causal_benchmark.json": {"passed": True, "metrics": {}},
        "phase25_agent_loop_benchmark.json": {"passed": True, "metrics": {}},
    }
    for name, payload in reports.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    journal = tmp_path / "journal.jsonl"
    baseline = module.build_review(str(tmp_path), journal_path=str(journal))
    journal.write_text(
        json.dumps({"metric_snapshot": baseline["metric_snapshot"]}) + "\n",
        encoding="utf-8",
    )
    reports["next_level_structural_benchmark.json"]["metrics"][
        "decision_accuracy"
    ] = 0.5
    (tmp_path / "next_level_structural_benchmark.json").write_text(
        json.dumps(reports["next_level_structural_benchmark.json"]),
        encoding="utf-8",
    )

    review = module.build_review(str(tmp_path), journal_path=str(journal))

    assert review["metric_drift"]["classification"] == (
        "nondeterministic_regression"
    )
    assert review["metric_drift"]["code_regression_detected"] is True
    assert review["checks"]["code_regression_absent"] is False
    assert review["promotion_allowed"] is False
