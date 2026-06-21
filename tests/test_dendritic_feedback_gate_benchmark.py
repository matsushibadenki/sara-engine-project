import json
import os
import subprocess
import sys


def test_dendritic_feedback_gate_benchmark_writes_managed_report():
    report_path = os.path.join("workspace", "evaluation", "test_dendritic_feedback_gate.json")
    summary_path = os.path.join("workspace", "evaluation", "test_dendritic_feedback_gate.txt")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval/dendritic_feedback_gate_benchmark.py",
            "--event-budget",
            "64",
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
    assert report["observed_only"] is True
    assert report["robustness_delta"] >= 0.0
    assert report["fallback_rate"] == 0.0
