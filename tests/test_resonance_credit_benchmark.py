import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_module():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "resonance_credit_benchmark.py")
    )
    spec = importlib.util.spec_from_file_location("resonance_credit_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resonance_credit_benchmark_writes_managed_evidence():
    module = _load_module()
    report_path = workspace_path("evaluation", "test_resonance_credit.json")
    exit_code = module.main(
        [
            "--fixture-path",
            processed_data_path("benchmark_fixtures", "test_resonance_credit.jsonl"),
            "--trace-path",
            workspace_path("evaluation", "test_resonance_credit_traces.jsonl"),
            "--state-path",
            workspace_path("evaluation", "test_resonance_credit_state.json"),
            "--report-path",
            report_path,
            "--summary-path",
            workspace_path("evaluation", "test_resonance_credit.txt"),
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["metrics"]["decision_integrity"] == 1.0
    assert report["metrics"]["harmful_update_suppression"] == 1.0
    assert report["metrics"]["naive_reward_harmful_update_count"] == 4
    assert report["metrics"]["resonance_update_count"] == 2
