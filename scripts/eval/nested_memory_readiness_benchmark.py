# Directory Path: scripts/eval/nested_memory_readiness_benchmark.py
# English Title: Nested Memory Readiness Benchmark
# Purpose/Content: Evaluates a lightweight Nested Learning-inspired continuum memory controller under CPU-only constraints.

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


def _load_module_from_path(module_name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_project_paths_helpers():
    module_path = os.path.join(PROJECT_ROOT, "src", "sara_engine", "utils", "project_paths.py")
    module = _load_module_from_path("sara_project_paths", module_path)
    return getattr(module, "ensure_parent_directory"), getattr(module, "workspace_path")


def _load_nested_memory_helpers():
    module_path = os.path.join(PROJECT_ROOT, "src", "sara_engine", "memory", "nested_continual.py")
    module = _load_module_from_path("sara_nested_continual", module_path)
    return getattr(module, "build_nested_memory_report")


ensure_parent_directory, workspace_path = _load_project_paths_helpers()
build_nested_memory_report = _load_nested_memory_helpers()

DEFAULT_REPORT_PATH = workspace_path("evaluation", "nested_memory_readiness_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "nested_memory_readiness_summary.txt")


def _benchmark_events() -> List[Dict[str, float]]:
    return [
        {"signal_strength": 0.82, "interference": 0.08, "novelty": 0.72, "urgency": 0.66},
        {"signal_strength": 0.88, "interference": 0.10, "novelty": 0.58, "urgency": 0.74},
        {"signal_strength": 0.76, "interference": 0.12, "novelty": 0.36, "urgency": 0.28},
        {"signal_strength": 0.91, "interference": 0.14, "novelty": 0.68, "urgency": 0.46},
        {"signal_strength": 0.94, "interference": 0.18, "novelty": 0.42, "urgency": 0.30},
        {"signal_strength": 0.78, "interference": 0.62, "novelty": 0.18, "urgency": 0.22},
        {"signal_strength": 0.87, "interference": 0.16, "novelty": 0.52, "urgency": 0.38},
        {"signal_strength": 0.90, "interference": 0.20, "novelty": 0.34, "urgency": 0.26},
        {"signal_strength": 0.84, "interference": 0.64, "novelty": 0.22, "urgency": 0.20},
        {"signal_strength": 0.93, "interference": 0.14, "novelty": 0.40, "urgency": 0.30},
        {"signal_strength": 0.89, "interference": 0.12, "novelty": 0.44, "urgency": 0.36},
        {"signal_strength": 0.95, "interference": 0.10, "novelty": 0.48, "urgency": 0.24},
        {"signal_strength": 0.92, "interference": 0.08, "novelty": 0.38, "urgency": 0.20},
        {"signal_strength": 0.96, "interference": 0.10, "novelty": 0.42, "urgency": 0.26},
        {"signal_strength": 0.94, "interference": 0.12, "novelty": 0.30, "urgency": 0.18},
        {"signal_strength": 0.97, "interference": 0.08, "novelty": 0.34, "urgency": 0.20},
    ]


def run_nested_memory_readiness_benchmark() -> Dict[str, Any]:
    report = build_nested_memory_report(_benchmark_events())
    metrics = report["metrics"]
    threshold_results = report["threshold_results"]
    return {
        "evaluator_name": "NestedMemoryReadinessBenchmark",
        "passed": bool(report["passed"]),
        "overall_score": sum(1.0 for passed in threshold_results.values() if passed) / max(len(threshold_results), 1),
        "metrics": metrics,
        "threshold_results": threshold_results,
        "details": report,
    }


def format_nested_memory_readiness_summary(report: Dict[str, Any]) -> str:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    threshold_results = (
        report.get("threshold_results", {})
        if isinstance(report.get("threshold_results"), dict)
        else {}
    )
    lines = [
        "SARA Engine Nested Memory Readiness Summary",
        f"- status: {'PASS' if bool(report.get('passed', False)) else 'FAIL'}",
        f"- overall_score: {float(report.get('overall_score', 0.0) or 0.0):.3f}",
    ]
    for name in sorted(threshold_results):
        lines.append(f"- {name}: {'PASS' if bool(threshold_results.get(name, False)) else 'FAIL'}")
    for name in sorted(metrics):
        lines.append(f"- metric.{name}: {float(metrics.get(name, 0.0) or 0.0):.3f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Nested Learning-inspired memory readiness benchmark.")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args()

    report = run_nested_memory_readiness_benchmark()
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_nested_memory_readiness_summary(report))

    print("Nested memory readiness benchmark completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
