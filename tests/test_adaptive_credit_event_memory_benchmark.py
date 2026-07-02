import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_module():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "adaptive_credit_event_memory_benchmark.py",
        )
    )
    spec = importlib.util.spec_from_file_location("adaptive_credit_event_memory_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adaptive_credit_event_memory_benchmark_writes_managed_evidence():
    module = _load_module()
    report_path = workspace_path("evaluation", "test_adaptive_credit_event_memory.json")
    exit_code = module.main(
        [
            "--fixture-path",
            processed_data_path("benchmark_fixtures", "test_adaptive_credit_event_memory.jsonl"),
            "--trace-path",
            workspace_path("evaluation", "test_adaptive_credit_event_memory_traces.jsonl"),
            "--report-path",
            report_path,
            "--summary-path",
            workspace_path("evaluation", "test_adaptive_credit_event_memory.txt"),
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["metrics"]["credit_strong_entry_present"] is True
    assert report["metrics"]["credit_weak_entry_evicted"] is True
    assert report["metrics"]["harmful_block_preserved_count"] >= 1
    assert report["metrics"]["bundle_longevity_bonus_present"] is True
