from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "eval"
        / "phase34_memory_checkpoint_cache_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase34_memory_checkpoint_cache_benchmark", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_phase34_benchmark_executes_without_promotion():
    module = _load_script()
    fixture = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_fixtures"
        / "phase34_memory_checkpoint_cache_cases.jsonl"
    )
    registration = (
        PROJECT_ROOT
        / "workspace"
        / "evaluation"
        / "phase34_memory_checkpoint_cache_preregistration.json"
    )
    rows = module.load_fixture(str(fixture))
    manifest = module.load_preregistration(str(registration))

    report = module.build_report(rows, manifest)

    assert report["execution_passed"] is True
    assert report["threshold_gate_passed"] is True
    assert report["mechanism_gate_passed"] is False
    assert report["promotion_ready"] is False
    assert report["metrics"]["condition_count"] == 64
    assert report["metrics"]["deterministic_replay"] == 1.0
    assert report["checks"]["no_durable_mutation"] is True
    assert report["checks"]["production_path_not_changed"] is True
    assert (
        report["arm_summaries"]["equal_segment_sparse_topk"][
            "delayed_recall_quality"
        ]
        > report["arm_summaries"]["recurrent_event_memory_control"][
            "delayed_recall_quality"
        ]
    )
    assert report["mechanism_observation"]["five_replicates_available"] is False
    assert (
        report["mechanism_observation"]["independent_evidence_available"]
        is False
    )


def test_benchmark_rejects_fixture_order_drift(tmp_path):
    module = _load_script()
    fixture = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_fixtures"
        / "phase34_memory_checkpoint_cache_cases.jsonl"
    )
    rows = module.load_fixture(str(fixture))
    rows.reverse()
    drifted = tmp_path / "drifted.jsonl"
    import json

    drifted.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    try:
        module.load_fixture(str(drifted))
    except ValueError as exc:
        assert "case_families" in str(exc)
    else:
        raise AssertionError("drifted fixture must not be accepted")
