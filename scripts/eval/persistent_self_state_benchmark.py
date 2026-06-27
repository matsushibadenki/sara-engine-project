from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Sequence

from sara_engine.dynamics import evaluate_persistent_self_state
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def build_report() -> Dict[str, Any]:
    evaluation = evaluate_persistent_self_state()
    metrics = evaluation.get("metrics", {})
    return {
        "schema": "sara-persistent-self-state-benchmark-v1",
        "observed_only": True,
        "passed": all(float(metrics.get(name, 0.0) or 0.0) >= 1.0 for name in (
            "persistent_self_state_idle_activity",
            "persistent_self_state_continuity",
            "persistent_self_state_memory_reactivation",
            "persistent_self_state_internal_prediction",
        )),
        "metrics": dict(metrics),
        "traces": dict(evaluation.get("traces", {})),
    }


def build_summary(report: Dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    lines = [
        "SARA Persistent self-state benchmark",
        f"- passed: {bool(report.get('passed', False))}",
        f"- idle_activity: {float(metrics.get('persistent_self_state_idle_activity', 0.0) or 0.0):.3f}",
        f"- continuity: {float(metrics.get('persistent_self_state_continuity', 0.0) or 0.0):.3f}",
        f"- memory_reactivation: {float(metrics.get('persistent_self_state_memory_reactivation', 0.0) or 0.0):.3f}",
        f"- internal_prediction: {float(metrics.get('persistent_self_state_internal_prediction', 0.0) or 0.0):.3f}",
    ]
    return "\n".join(lines) + "\n"


DEFAULT_REPORT_PATH = workspace_path("evaluation", "persistent_self_state_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path(
    "evaluation",
    "persistent_self_state_benchmark_summary.txt",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the persistent self-state benchmark.")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def run_benchmark(
    *,
    report_path: str = DEFAULT_REPORT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
) -> Dict[str, Any]:
    report = build_report()
    summary = build_summary(report)
    ensure_parent_directory(report_path)
    ensure_parent_directory(summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summary)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(report_path=args.report_path, summary_path=args.summary_path)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if bool(report.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
