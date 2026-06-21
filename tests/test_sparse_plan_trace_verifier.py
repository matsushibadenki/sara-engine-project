import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_verifier_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "sparse_plan_trace_verifier.py")
    )
    spec = importlib.util.spec_from_file_location("sparse_plan_trace_verifier", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sparse_plan_trace_verifier_writes_managed_outputs():
    verifier = _load_verifier_module()
    fixture_path = processed_data_path("benchmark_fixtures", "test_sparse_plan_trace_cases.jsonl")
    repair_path = processed_data_path("autobot", "test_plan_trace_repair_materials.jsonl")
    report_path = workspace_path("evaluation", "test_sparse_plan_trace_verifier.json")
    summary_path = workspace_path("evaluation", "test_sparse_plan_trace_verifier.txt")

    exit_code = verifier.main(
        [
            "--fixture-path",
            fixture_path,
            "--repair-path",
            repair_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
        ]
    )

    assert exit_code == 0
    assert os.path.exists(fixture_path)
    assert os.path.exists(repair_path)
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["observed_only"] is True
    assert report["case_count"] >= 5
    assert report["repair_material_count"] >= 1
    assert report["invalid_step_count"] >= 1
