from __future__ import annotations

import importlib.util
from pathlib import Path

from sara_engine.evaluation.phase33_preregistration import (
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


def test_phase33_benchmark_executes_every_registered_condition_without_promotion():
    draft_module = _load_script("phase33_structured_edge_draft")
    benchmark_module = _load_script("phase33_structured_edge_benchmark")
    fixture = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_fixtures"
        / "phase33_structured_edge_cases.jsonl"
    )
    rows = benchmark_module.load_fixture(str(fixture))
    draft = draft_module.build_draft(
        rows,
        draft_module.environment_descriptor(),
    )
    manifest = build_registered_manifest(draft, managed_path=True)

    report = benchmark_module.build_report(rows, manifest)

    assert report["execution_passed"] is True
    assert report["promotion_ready"] is False
    assert report["metrics"]["condition_count"] == 1275
    assert report["checks"]["deterministic_replay"] is True
    assert report["checks"]["no_durable_mutation"] is True
    assert report["mechanism_observation"][
        "simplification_evidence_available"
    ] is False
    assert report["mechanism_observation"][
        "independent_evidence_available"
    ] is False
    assert report["mechanism_observation"][
        "linear_multi_beats_single_scalar"
    ] is False
    assert (
        report["arm_summaries"]["branch_local_contacts"][
            "ambiguous_relation_quality"
        ]
        > report["arm_summaries"]["typed_independent_contacts"][
            "ambiguous_relation_quality"
        ]
    )
