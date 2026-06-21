import json
import os
import subprocess
import sys


def test_operator_llm_assistant_readiness_writes_managed_outputs(tmp_path):
    report_path = os.path.join("workspace", "evaluation", "test_operator_llm_readiness.json")
    summary_path = os.path.join("workspace", "evaluation", "test_operator_llm_readiness.txt")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval/operator_llm_assistant_readiness.py",
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert os.path.exists(report_path)
    assert os.path.exists(summary_path)
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["disabled_by_default"] is True
    assert report["llm_runtime_required"] is False
    assert report["accepted_count"] == 2
    assert report["rejected_count"] == 7
    assert report["rejection_counts"]["direct_mutation_action"] == 2
    assert report["rejection_counts"]["unmanaged_output_path"] == 1
    assert report["rejection_counts"]["secret_like_text"] == 1
