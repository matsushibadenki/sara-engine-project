from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "eval"
        / "phase34_memory_cache_factorial_independent_adapter_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("phase34_independent_adapter_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(module):
    return (
        module._read_jsonl(module.DEFAULT_MANIFEST),
        module._read_json(module.DEFAULT_CASE_PLAN),
        module._read_json(module.DEFAULT_PREREGISTRATION),
        module._read_json(module.DEFAULT_PARENT_REPORT),
        module._read_json(module.DEFAULT_EXTERNAL_GATE),
        module._read_json(module.DEFAULT_READINESS_GATE),
    )


def test_registered_independent_adapter_executes_all_source_identity_conditions():
    module = _load_script()
    report = module.build_report(*_inputs(module))

    assert report["execution_passed"] is True
    assert report["identity_gate_passed"] is True
    assert report["promotion_ready"] is False
    assert report["semantic_accuracy_claim_allowed"] is False
    assert report["metrics"]["condition_count"] == 1050
    assert report["metrics"]["planned_unique_material_count"] == 66
    assert report["metrics"]["planned_unique_source_ref_count"] == 66
    assert report["metrics"]["retained_set_identity"] == 1.0
    assert report["metrics"]["selection_precision_main_effect"] == 0.875
    assert report["metrics"]["selection_recall_noninferiority"] == 0.0
    assert report["metrics"]["retention_old_recall_main_effect"] == 1.0
    assert report["metrics"]["positive_identity_recall_by_arm"][
        "logarithmic_retention_sparse_topk"
    ] == 1.0
    assert all(report["checks"].values())
    assert all(report["metric_gates"].values())


def test_independent_adapter_rejects_case_plan_drift_before_execution():
    module = _load_script()
    rows, plan, preregistration, parent, external, readiness = _inputs(module)
    plan = copy.deepcopy(plan)
    plan["cases"][0]["stream_material_hashes"].reverse()

    with pytest.raises(ValueError, match="case_plan_fingerprint_matches"):
        module.build_report(rows, plan, preregistration, parent, external, readiness)


def test_independent_runtime_case_binds_material_hashes_to_source_refs():
    module = _load_script()
    rows, plan, *_ = _inputs(module)
    source_by_hash = {row["material_hash"]: row for row in rows}
    runtime_case = module.adapt_case(plan["cases"][0], source_by_hash)

    assert len(runtime_case["checkpoint_stream"]) == len(runtime_case["checkpoint_source_refs"])
    assert runtime_case["query_ids"][0].startswith("target:")
    assert runtime_case["semantic_accuracy_claim_allowed"] is False
