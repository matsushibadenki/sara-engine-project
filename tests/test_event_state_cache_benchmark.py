import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import (
    interim_data_path,
    processed_data_path,
    workspace_path,
)


def _load_module():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "event_state_cache_benchmark.py",
        )
    )
    spec = importlib.util.spec_from_file_location(
        "event_state_cache_benchmark",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_event_state_cache_benchmark_writes_managed_evidence():
    module = _load_module()
    report_path = workspace_path("evaluation", "test_event_state_cache.json")
    exit_code = module.main(
        [
            "--fixture-path",
            processed_data_path(
                "benchmark_fixtures",
                "test_event_state_cache.jsonl",
            ),
            "--candidate-path",
            interim_data_path(
                "event_state_cache",
                "test_candidates.jsonl",
            ),
            "--manifest-path",
            processed_data_path(
                "event_state_cache",
                "test_manifest.jsonl",
            ),
            "--trace-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache_traces.jsonl",
            ),
            "--state-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache_state.json",
            ),
            "--report-path",
            report_path,
            "--summary-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache.txt",
            ),
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["observed_only"] is True
    assert report["metrics"]["logarithmic_delayed_recall"] == 1.0
    assert report["metrics"]["fixed_delayed_recall"] < 1.0
    assert report["metrics"]["blocked_decision_integrity"] == 1.0
    assert (
        report["metrics"]["logarithmic_entry_count"]
        < report["metrics"]["linear_entry_count"]
    )
