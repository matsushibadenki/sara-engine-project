#!/usr/bin/env python3
"""Validate the observed-only Phase 15 sparse dendritic feedback surface."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


DEFAULT_BENCHMARK_PATH = workspace_path("evaluation", "dendritic_feedback_gate_benchmark.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase15_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase15_completion_gate_summary.txt")
MAX_EVENT_BUDGET = 256
MAX_STATE_BUDGET = 256


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_report(*, benchmark_path: str = DEFAULT_BENCHMARK_PATH) -> Dict[str, Any]:
    benchmark = _load_json(benchmark_path)
    notes = benchmark.get("policy_notes", []) if isinstance(benchmark, Mapping) else []
    policy_text = " ".join(str(item).lower() for item in notes)
    rows = benchmark.get("rows", []) if isinstance(benchmark, Mapping) else []
    row_list = rows if isinstance(rows, list) else []
    checks = {
        "benchmark_present": benchmark is not None,
        "benchmark_passed": bool(benchmark and benchmark.get("passed")),
        "benchmark_observed_only": bool(benchmark and benchmark.get("observed_only")),
        "case_coverage": len(row_list) >= 4,
        "robustness_non_negative": float(benchmark.get("robustness_delta", -1.0) or -1.0) >= 0.0 if benchmark else False,
        "traceability_visible": all(isinstance(row, Mapping) and row.get("case_id") and row.get("convergence_steps") is not None for row in row_list),
        "event_cost_bounded": int(benchmark.get("max_event_cost", MAX_EVENT_BUDGET + 1) or 0) <= MAX_EVENT_BUDGET if benchmark else False,
        "state_budget_bounded": int(benchmark.get("max_state_budget_units", MAX_STATE_BUDGET + 1) or 0) <= MAX_STATE_BUDGET if benchmark else False,
        "fallback_visible": benchmark is not None and benchmark.get("fallback_rate") is not None,
        "bounded_sparse_policy_visible": all(term in policy_text for term in ("sparse", "cpu-first", "bounded-state", "backpropagation-free")),
        "production_path_unchanged": "does not alter default production inference" in policy_text,
    }
    passed = all(checks.values())
    next_actions: List[Dict[str, Any]] = []
    if passed:
        next_actions.append({"priority": 3, "reason": "keep dendritic evidence observed-only until repeated robustness and cost review", "command": "python scripts/sara_cli.py eval-operator-dashboard"})
    else:
        next_actions.append({"priority": 1, "reason": "refresh_or_review_phase15_evidence", "command": "python scripts/sara_cli.py eval-research-benchmark-suite"})
    return {
        "schema": "sara-phase15-completion-gate-v1",
        "phase": 15,
        "phase15_complete": passed,
        "status": "phase15_complete" if passed else "phase15_incomplete",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "case_count": int(benchmark.get("case_count", 0) or 0) if benchmark else 0,
            "robustness_delta": float(benchmark.get("robustness_delta", 0.0) or 0.0) if benchmark else 0.0,
            "fallback_rate": float(benchmark.get("fallback_rate", 0.0) or 0.0) if benchmark else 0.0,
            "max_event_cost": int(benchmark.get("max_event_cost", 0) or 0) if benchmark else 0,
            "max_state_budget_units": int(benchmark.get("max_state_budget_units", 0) or 0) if benchmark else 0,
        },
        "evidence_path": benchmark_path,
        "next_actions": next_actions,
        "promotion_rule": {"release_critical": False, "observed_only_until_stable": True, "requires_robustness_cost_fallback_regression_review": True},
        "what_is_not_proven": [
            "Phase 15 fixture evidence does not prove broad external generalization.",
            "Proxy event cost does not prove physical joule-per-success advantage.",
            "Dendritic feedback is not promoted to release-critical default inference by this gate alone.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 15 sparse dendritic feedback evidence.")
    parser.add_argument("--benchmark-path", default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(benchmark_path=args.benchmark_path)
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    lines = ["Phase 15 completion gate", f"status: {report['status']}", f"phase15_complete: {str(report['phase15_complete']).lower()}"]
    lines.extend(f"- {name}: {str(value).lower()}" for name, value in report["checks"].items())
    lines.append("Next actions:")
    lines.extend(f"- {item['reason']} -> {item['command']}" for item in report["next_actions"])
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
