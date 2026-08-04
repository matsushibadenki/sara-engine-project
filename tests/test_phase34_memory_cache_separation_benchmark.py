from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "eval"
        / "phase34_memory_cache_separation_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase34_memory_cache_separation_benchmark", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_separation_followup_executes_and_preserves_negative_result():
    module = _load_script()
    rows = module.load_fixture(
        str(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "benchmark_fixtures"
            / "phase34_memory_cache_separation_cases.jsonl"
        )
    )
    manifest = module.load_preregistration(
        str(
            PROJECT_ROOT
            / "workspace"
            / "evaluation"
            / "phase34_memory_cache_separation_preregistration.json"
        )
    )

    report = module.build_report(rows, manifest)

    assert report["execution_passed"] is True
    assert report["threshold_gate_passed"] is False
    assert report["mechanism_gate_passed"] is False
    assert report["promotion_ready"] is False
    assert report["metrics"]["condition_count"] == 240
    assert report["metrics"]["deterministic_replay"] == 1.0
    assert report["metrics"]["logarithmic_old_recall_delta"] == 1.0
    assert report["metrics"]["topk_pollution_precision_delta"] == 0.0
    assert report["metric_gates"]["topk_pollution_precision_delta"] is False
    assert report["failure_analysis"]["topk_retention_selection_confound"] is True
    assert report["failure_analysis"]["registration_mutated"] is False
    assert report["checks"]["all_five_registered_seeds_executed"] is True
    assert report["checks"]["no_durable_mutation"] is True
