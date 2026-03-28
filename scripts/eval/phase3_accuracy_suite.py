# Directory Path: scripts/eval/phase3_accuracy_suite.py
# English Title: Phase 3 Accuracy Suite
# Purpose/Content: Aggregates lightweight Phase 3 benchmarks for SaraAgent, SaraInference, and SpikingLLM into a managed report under workspace/.

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, Optional


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
from sara_engine.evaluation.phase3_tracking import (
    append_phase3_history,
    build_phase3_trend,
    latest_phase3_report,
)
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _status_label(passed: bool) -> str:
    return "PASS" if passed else "WARN"


def _build_focus_summary(component_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    inference_metrics = component_reports.get("sara_inference", {}).get("metrics", {})
    llm_metrics = component_reports.get("spiking_llm", {}).get("metrics", {})

    few_shot_metrics = {
        "sara_inference.few_shot_accuracy": float(inference_metrics.get("few_shot_accuracy", 0.0)),
        "spiking_llm.few_shot_context_accuracy": float(llm_metrics.get("few_shot_context_accuracy", 0.0)),
    }
    continual_metrics = {
        "sara_inference.continual_retention": float(inference_metrics.get("continual_retention", 0.0)),
        "sara_inference.long_horizon_retention": float(inference_metrics.get("long_horizon_retention", 0.0)),
        "spiking_llm.continual_memory_retention": float(llm_metrics.get("continual_memory_retention", 0.0)),
        "spiking_llm.long_horizon_memory_retention": float(llm_metrics.get("long_horizon_memory_retention", 0.0)),
    }

    few_shot_score = sum(few_shot_metrics.values()) / max(len(few_shot_metrics), 1)
    continual_score = sum(continual_metrics.values()) / max(len(continual_metrics), 1)
    return {
        "few_shot": {
            "score": few_shot_score,
            "passed": all(value >= 1.0 for value in few_shot_metrics.values()),
            "metrics": few_shot_metrics,
        },
        "continual": {
            "score": continual_score,
            "passed": all(value >= 1.0 for value in continual_metrics.values()),
            "metrics": continual_metrics,
        },
    }


def format_phase3_accuracy_summary(report: Dict[str, Any]) -> str:
    focus_summary = report.get("focus_summary", {}) if isinstance(report.get("focus_summary"), dict) else {}
    trend = report.get("trend", {}) if isinstance(report.get("trend"), dict) else {}
    few_shot = focus_summary.get("few_shot", {}) if isinstance(focus_summary.get("few_shot"), dict) else {}
    continual = focus_summary.get("continual", {}) if isinstance(focus_summary.get("continual"), dict) else {}

    lines = [
        "SARA Engine Phase 3 Accuracy Summary",
        f"overall_status: {_status_label(bool(report.get('passed', False)))}",
        f"overall_score: {float(report.get('overall_score', 0.0)):.3f}",
        f"regression_count: {int(trend.get('regression_count', 0))}",
        "",
        "Focus",
        f"- few_shot_status: {_status_label(bool(few_shot.get('passed', False)))}",
        f"- few_shot_score: {float(few_shot.get('score', 0.0)):.3f}",
        f"- continual_status: {_status_label(bool(continual.get('passed', False)))}",
        f"- continual_score: {float(continual.get('score', 0.0)):.3f}",
        "",
        "Components",
    ]

    component_reports = report.get("component_reports", {})
    if isinstance(component_reports, dict):
        for component_name, component_report in sorted(component_reports.items()):
            if not isinstance(component_report, dict):
                continue
            lines.append(
                f"- {component_name}: {_status_label(bool(component_report.get('passed', False)))} "
                f"score={float(component_report.get('overall_score', 0.0)):.3f}"
            )

    return "\n".join(lines) + "\n"


def run_phase3_accuracy_suite(
    history_path: Optional[str] = None,
    persist_history: bool = False,
    history_limit: int = 50,
) -> Dict[str, Any]:
    benchmarks: Dict[str, Callable[[], Dict[str, Any]]] = {
        "agent_dialogue": run_agent_dialogue_benchmark,
        "sara_inference": run_inference_accuracy_benchmark,
        "spiking_llm": run_spiking_llm_accuracy_benchmark,
    }
    reports = {name: benchmark() for name, benchmark in benchmarks.items()}
    overall_score = sum(report["overall_score"] for report in reports.values()) / max(len(reports), 1)
    passed = all(bool(report.get("passed", False)) for report in reports.values())

    previous_report = latest_phase3_report(history_path) if history_path else None
    focus_summary = _build_focus_summary(reports)
    report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": overall_score,
        "component_reports": reports,
        "focus_summary": focus_summary,
        "passed": passed,
        "trend": build_phase3_trend(
            current_report={
                "suite_name": "Phase3AccuracySuite",
                "overall_score": overall_score,
                "component_reports": reports,
                "focus_summary": focus_summary,
                "passed": passed,
            },
            previous_report=previous_report,
        ),
    }
    if previous_report is not None:
        report["previous_overall_score"] = previous_report.get("overall_score")

    if history_path and persist_history:
        history = append_phase3_history(
            history_path=history_path,
            report=report,
            max_entries=history_limit,
        )
        report["history_length"] = len(history)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the aggregated Phase 3 accuracy suite.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase3_accuracy_suite.json"),
        help="Managed output path for the aggregated report.",
    )
    parser.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase3_accuracy_summary.txt"),
        help="Managed output path for the human-readable accuracy summary.",
    )
    parser.add_argument(
        "--history-path",
        default=workspace_path("evaluation", "phase3_accuracy_history.json"),
        help="Managed output path for suite history snapshots.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=50,
        help="Maximum number of suite snapshots to keep in history.",
    )
    args = parser.parse_args()

    report = run_phase3_accuracy_suite(
        history_path=args.history_path,
        persist_history=True,
        history_limit=args.history_limit,
    )
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    summary_path = ensure_parent_directory(args.summary_path)
    summary_text = format_phase3_accuracy_summary(report)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summary_text)

    print("Phase 3 accuracy suite completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
