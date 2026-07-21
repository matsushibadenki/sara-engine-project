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
            "independent_source_scope": {"domains": 2},
        },
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
