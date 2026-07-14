#!/usr/bin/env python3
"""Validate the observed-only Phase 16 sparse multimodal binding surface."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


DEFAULT_BENCHMARK_PATH = workspace_path("evaluation", "synesthetic_multimodal_binding_benchmark.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase16_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase16_completion_gate_summary.txt")
MAX_EVENT_COST = 256
MAX_STATE_BUDGET = 256
REQUIRED_METRICS = (
    "adapter_ir_integrity",
    "temporal_alignment_quality",
    "cross_modal_link_precision",
    "plug_swap_integrity",
    "missing_modality_abstention_integrity",
    "non_language_route_usefulness",
    "bundle_integrity",
    "binding_audit_coverage",
    "route_traceability",
)


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
    missing_results = benchmark.get("missing_modality_results", []) if isinstance(benchmark, Mapping) else []
    missing_rows = missing_results if isinstance(missing_results, list) else []
    windows = benchmark.get("window_profiles", []) if isinstance(benchmark, Mapping) else []
    window_values = {float(row.get("window_ms")) for row in windows if isinstance(row, Mapping) and row.get("window_ms") is not None}
    has_prediction = any(
        isinstance(row, Mapping)
        and row.get("observed") is False
        and isinstance(row.get("predicted_missing_modality_events"), list)
        for row in missing_rows
    )
    has_abstention = any(
        isinstance(row, Mapping)
        and row.get("abstained") is True
        and float(row.get("uncertainty", 0.0) or 0.0) > 0.0
        for row in missing_rows
    )
    checks = {
        "benchmark_present": benchmark is not None,
        "benchmark_passed": bool(benchmark and benchmark.get("passed")),
        "benchmark_observed_only": bool(benchmark and benchmark.get("observed_only")),
        "case_coverage": int(benchmark.get("case_count", 0) or 0) >= 4 if benchmark else False,
        "required_metrics_passed": all(float(metrics.get(name, 0.0) or 0.0) >= 1.0 for name in REQUIRED_METRICS),
        "temporal_profiles_visible": window_values == {25.0, 32.0, 40.0},
        "selected_window_visible": float(benchmark.get("selected_window_ms", 0.0) or 0.0) in window_values if benchmark else False,
        "missing_modality_prediction_labeled": has_prediction,
        "missing_modality_abstention_visible": has_abstention,
        "event_cost_bounded": int(metrics.get("max_event_cost", MAX_EVENT_COST + 1) or 0) <= MAX_EVENT_COST,
        "state_budget_bounded": int(metrics.get("max_state_budget_units", MAX_STATE_BUDGET + 1) or 0) <= MAX_STATE_BUDGET,
        "sparse_equal_modality_policy_visible": "sparse events" in policy_text and "dense universal" in policy_text and "bounded" in policy_text,
        "separable_bundle_policy_visible": "preserve modality-local payloads" in policy_text,
    }
    passed = all(checks.values())
    next_actions: List[Dict[str, Any]] = []
    if passed:
        next_actions.append({"priority": 3, "reason": "keep multimodal binding observed-only until repeated quality, abstention, and cost review", "command": "python scripts/sara_cli.py eval-operator-dashboard"})
    else:
        next_actions.append({"priority": 1, "reason": "refresh_or_review_phase16_evidence", "command": "python scripts/sara_cli.py eval-research-benchmark-suite"})
    return {
        "schema": "sara-phase16-completion-gate-v1",
        "phase": 16,
        "phase16_complete": passed,
        "status": "phase16_complete" if passed else "phase16_incomplete",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "case_count": int(benchmark.get("case_count", 0) or 0) if benchmark else 0,
            "selected_window_ms": float(benchmark.get("selected_window_ms", 0.0) or 0.0) if benchmark else 0.0,
            "max_event_cost": int(metrics.get("max_event_cost", 0) or 0),
            "max_state_budget_units": int(metrics.get("max_state_budget_units", 0) or 0),
            "missing_modality_case_count": len(missing_rows),
        },
        "evidence_path": benchmark_path,
        "next_actions": next_actions,
        "promotion_rule": {"release_critical": False, "observed_only_until_stable": True, "requires_quality_abstention_energy_trace_regression_review": True},
        "what_is_not_proven": [
            "Phase 16 fixture evidence does not prove broad real-world multimodal generalization.",
            "Proxy event cost does not prove physical joule-per-success advantage.",
            "Multimodal binding is not promoted to release-critical runtime behavior by this gate alone.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 16 sparse synesthetic multimodal binding evidence.")
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
    lines = ["Phase 16 completion gate", f"status: {report['status']}", f"phase16_complete: {str(report['phase16_complete']).lower()}"]
    lines.extend(f"- {name}: {str(value).lower()}" for name, value in report["checks"].items())
    lines.append("Next actions:")
    lines.extend(f"- {item['reason']} -> {item['command']}" for item in report["next_actions"])
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
