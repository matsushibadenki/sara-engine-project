from __future__ import annotations

import importlib.util
from pathlib import Path

from sara_engine.evaluation.phase33_twinprop_preregistration import (
    build_registered_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_twinprop_ablation_executes_without_production_promotion():
    draft_module = _load_script("phase33_twinprop_ablation_draft")
    benchmark_module = _load_script("phase33_twinprop_ablation_benchmark")
    fixture = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_fixtures"
        / "phase33_twinprop_ablation_cases.jsonl"
    )
    rows = benchmark_module.load_fixture(str(fixture))
    manifest = build_registered_manifest(
        draft_module.build_draft(
            rows,
            draft_module.environment_descriptor(),
        ),
        managed_path=True,
    )

    report = benchmark_module.build_report(rows, manifest)

    assert report["execution_passed"] is True
    assert report["mechanism_gate_passed"] is True
    assert report["promotion_ready"] is False
    assert report["independent_evidence_available"] is False
    assert report["metrics"]["condition_count"] == 350
    assert report["metrics"]["deterministic_replay"] == 1.0
    assert report["metrics"]["branch_participation_monotonicity"] == 1.0
    assert report["metrics"]["structured_over_shuffled_delta"] >= 0.1
    assert report["checks"]["same_fixed_readout_across_arms"] is True
    assert report["checks"]["no_durable_mutation"] is True
    assert report["checks"]["dense_digital_twin_not_used"] is True
    assert (
        report["arm_summaries"]["intact_bounded_branches"][
            "fixed_readout_quality"
        ]
        > report["arm_summaries"]["passive_linear_branches"][
            "fixed_readout_quality"
        ]
    )
