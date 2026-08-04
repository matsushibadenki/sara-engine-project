from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "eval"
        / "phase34_memory_cache_factorial_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase34_memory_cache_factorial_benchmark", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_factorial_executes_with_identifiable_topk_effect():
    module = _load_script()
    rows = module.load_fixture(
        str(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "benchmark_fixtures"
            / "phase34_memory_cache_factorial_cases.jsonl"
        )
    )
    manifest = module.load_preregistration(
        str(
            PROJECT_ROOT
            / "workspace"
            / "evaluation"
            / "phase34_memory_cache_factorial_preregistration.json"
        )
    )

    report = module.build_report(rows, manifest)

    assert report["execution_passed"] is True
    assert report["threshold_gate_passed"] is True
    assert report["mechanism_gate_passed"] is True
    assert report["promotion_ready"] is False
    assert report["metrics"]["condition_count"] == 300
    assert report["metrics"]["retained_set_identity"] == 1.0
    assert report["metrics"]["selection_precision_main_effect"] == 0.875
    assert report["metrics"]["selection_recall_noninferiority"] == 0.0
    assert report["metrics"]["selection_retention_interaction_abs"] == 0.0
    assert report["metrics"]["retention_old_recall_main_effect"] == 1.0
    assert report["checks"]["retention_query_blind"] is True
    assert report["checks"]["no_durable_mutation"] is True
    assert report["independent_evidence_available"] is False
