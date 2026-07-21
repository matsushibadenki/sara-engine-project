from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "level2_capability_matrix.py"
    spec = importlib.util.spec_from_file_location("level2_capability_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_level2_matrix_keeps_promotion_blocked_for_missing_independent_evidence():
    module = _load_module()
    report = module.build_matrix("/nonexistent/evaluation")

    assert report["promotion_allowed"] is False
    assert report["checks"]["human_review_gate_pass"] is False
    assert "independent multimodal workload" in report["unresolved_gaps"]


def test_level2_matrix_reports_internal_capabilities_from_managed_reports(tmp_path):
    module = _load_module()
    payloads = {
        "next_level_structural_benchmark.json": {"passed": True, "metrics": {"supported_composition": 1.0, "unsupported_composition_abstention": 1.0}},
        "continual_horizon_benchmark.json": {"passed": True, "metrics": {"mean_active_useful_recall": 1.0, "mean_active_protected_knowledge_retention": 1.0, "max_state_growth": 4}},
        "phase23_structural_fusion_benchmark.json": {"passed": True, "metrics": {"decision_accuracy": 1.0, "contradiction_abstention": 1.0}},
        "phase24_causal_benchmark.json": {"passed": True, "metrics": {"verified_causal_case": 1.0}, "checks": {"unsupported_counterfactual_abstention": True}},
        "phase25_agent_loop_benchmark.json": {"passed": True, "metrics": {"safe_plan_acceptance": 1.0}, "checks": {"unexpected_outcome_rolls_back": True}},
        "continual_horizon_external_gate.json": {"promotion_allowed": False},
        "next_level_promotion_gate.json": {"promotion_allowed": False},
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(__import__("json").dumps(payload), encoding="utf-8")

    report = module.build_matrix(str(tmp_path))

    assert report["checks"]["internal_capabilities_pass"] is True
    assert report["promotion_allowed"] is False


def test_level2_matrix_has_a_real_promotion_transition(tmp_path):
    module = _load_module()
    payloads = {
        "next_level_structural_benchmark.json": {"passed": True},
        "continual_horizon_benchmark.json": {"passed": True},
        "phase23_structural_fusion_benchmark.json": {
            "passed": True,
            "independent_source_scope": {"domains": 2},
        },
        "phase24_causal_benchmark.json": {"passed": True},
        "phase25_agent_loop_benchmark.json": {"passed": True},
        "continual_horizon_external_gate.json": {"promotion_allowed": True},
        "next_level_promotion_gate.json": {"promotion_allowed": True},
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(__import__("json").dumps(payload), encoding="utf-8")

    report = module.build_matrix(str(tmp_path))

    assert report["unresolved_gaps"] == []
    assert report["promotion_allowed"] is True
