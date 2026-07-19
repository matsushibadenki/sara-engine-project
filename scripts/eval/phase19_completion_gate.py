#!/usr/bin/env python3
"""Validate the observed-only Phase 19 sparse liquid-time-constant surface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


DEFAULT_BENCHMARK_PATH = workspace_path("evaluation", "sparse_liquid_time_constant_benchmark.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase19_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase19_completion_gate_summary.txt")
MAX_EVENT_COST = 64
MAX_UPDATE_COUNT = 3
MAX_STATE_BUDGET = 32
MAX_TAU = 64.0


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_report(*, benchmark_path: str = DEFAULT_BENCHMARK_PATH) -> Dict[str, Any]:
    benchmark = _load_json(benchmark_path)
    metrics = benchmark.get("metrics", {}) if isinstance(benchmark, Mapping) else {}
    notes = benchmark.get("policy_notes", []) if isinstance(benchmark, Mapping) else []
    policy_text = " ".join(str(item).lower() for item in notes)
    checks = {
        "benchmark_present": benchmark is not None,
        "benchmark_passed": bool(benchmark and benchmark.get("passed")),
        "benchmark_observed_only": bool(benchmark and benchmark.get("observed_only")),
        "case_coverage": int(benchmark.get("case_count", 0) or 0) >= 4 if benchmark else False,
        "improves_fixed_control": float(metrics.get("liquid_improves_fixed", 0.0) or 0.0) >= 1.0,
        "improves_multiscale_control": float(metrics.get("liquid_improves_multiscale", 0.0) or 0.0) >= 1.0,
        "replay_determinism": float(metrics.get("replay_determinism", 0.0) or 0.0) >= 1.0,
        "abstention_integrity": float(metrics.get("abstention_integrity", 0.0) or 0.0) >= 1.0,
        "event_cost_bounded": int(metrics.get("max_event_cost", MAX_EVENT_COST + 1) or 0) <= MAX_EVENT_COST,
        "update_count_bounded": int(metrics.get("max_update_count", MAX_UPDATE_COUNT + 1) or 0) <= MAX_UPDATE_COUNT,
        "state_budget_bounded": int(metrics.get("max_state_budget_units", MAX_STATE_BUDGET + 1) or 0) <= MAX_STATE_BUDGET,
        "time_constant_bounded": float(metrics.get("max_time_constant", MAX_TAU + 1.0) or 0.0) <= MAX_TAU,
        "sparse_cpu_policy_visible": all(term in policy_text for term in ("sparse", "cpu", "closed-form", "backpropagation")),
        "production_control_preserved": "fixed-time-constant snn remains the default" in policy_text and "does not alter production" in policy_text,
    }
    passed = all(checks.values())
    next_actions: List[Dict[str, Any]] = []
    if passed:
        next_actions.append({"priority": 3, "reason": "keep liquid dynamics observed-only until held-out quality, energy, and regression review", "command": "python scripts/sara_cli.py eval-operator-dashboard"})
    else:
        next_actions.append({"priority": 1, "reason": "refresh_or_review_phase19_evidence", "command": "python scripts/sara_cli.py eval-research-benchmark-suite"})
    return {
        "schema": "sara-phase19-completion-gate-v1",
        "phase": 19,
        "phase19_complete": passed,
        "status": "phase19_complete" if passed else "phase19_incomplete",
        "passed": passed,
        "checks": checks,
        "metrics": metrics,
        "evidence_path": benchmark_path,
        "next_actions": next_actions,
        "promotion_rule": {"release_critical": False, "observed_only_until_stable": True, "requires_held_out_quality_energy_latency_regression_review": True},
        "what_is_not_proven": [
            "Phase 19 fixture evidence does not prove independent held-out real-world temporal generalization.",
            "Proxy event cost does not prove physical joule-per-success advantage.",
            "Liquid dynamics are not promoted to release-critical default inference by this gate alone.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 19 sparse liquid-time-constant evidence.")
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
    lines = ["Phase 19 completion gate", f"status: {report['status']}", f"phase19_complete: {str(report['phase19_complete']).lower()}"]
    lines.extend(f"- {name}: {str(value).lower()}" for name, value in report["checks"].items())
    lines.append("Next actions:")
    lines.extend(f"- {item['reason']} -> {item['command']}" for item in report["next_actions"])
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
