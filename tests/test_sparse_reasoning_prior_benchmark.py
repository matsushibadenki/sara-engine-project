import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_module():
    module_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "sparse_reasoning_prior_benchmark.py",
        )
    )
    spec = importlib.util.spec_from_file_location("sparse_reasoning_prior_benchmark", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sparse_reasoning_prior_benchmark_writes_managed_outputs():
    module = _load_module()
    fixture_path = processed_data_path(
        "benchmark_fixtures", "test_sparse_reasoning_prior_cases.jsonl"
    )
    trace_path = workspace_path("evaluation", "test_sparse_reasoning_prior_traces.jsonl")
    report_path = workspace_path("evaluation", "test_sparse_reasoning_prior.json")
    summary_path = workspace_path("evaluation", "test_sparse_reasoning_prior.txt")

    exit_code = module.main(
        [
            "--fixture-path",
            fixture_path,
            "--trace-path",
            trace_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["observed_only"] is True
    assert report["case_count"] == 5
    assert report["metrics"]["logic_to_state_consistency"] == 1.0
    assert report["metrics"]["external_event_missing_abstention"] == 1.0
    assert report["metrics"]["source_backed_integrity"] == 1.0
