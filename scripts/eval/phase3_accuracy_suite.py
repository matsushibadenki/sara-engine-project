# Directory Path: scripts/eval/phase3_accuracy_suite.py
# English Title: Phase 3 Accuracy Suite
# Purpose/Content: Aggregates lightweight Phase 3 benchmarks for SaraAgent, SaraInference, and SpikingLLM into a managed report under workspace/.

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_PATH = os.path.dirname(__file__)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SCRIPT_PATH not in sys.path:
    sys.path.insert(0, SCRIPT_PATH)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from agent_dialogue_benchmark import run_agent_dialogue_benchmark
from inference_accuracy_benchmark import run_inference_accuracy_benchmark
from spiking_llm_accuracy_benchmark import run_spiking_llm_accuracy_benchmark
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def run_phase3_accuracy_suite() -> Dict[str, Any]:
    benchmarks: Dict[str, Callable[[], Dict[str, Any]]] = {
        "agent_dialogue": run_agent_dialogue_benchmark,
        "sara_inference": run_inference_accuracy_benchmark,
        "spiking_llm": run_spiking_llm_accuracy_benchmark,
    }
    reports = {name: benchmark() for name, benchmark in benchmarks.items()}
    overall_score = sum(report["overall_score"] for report in reports.values()) / max(len(reports), 1)
    passed = all(bool(report.get("passed", False)) for report in reports.values())
    return {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": overall_score,
        "component_reports": reports,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the aggregated Phase 3 accuracy suite.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase3_accuracy_suite.json"),
        help="Managed output path for the aggregated report.",
    )
    args = parser.parse_args()

    report = run_phase3_accuracy_suite()
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("Phase 3 accuracy suite completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
