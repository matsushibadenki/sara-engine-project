#!/usr/bin/env python3
"""Validate the observed-only Phase 18 verified hierarchical event-state cache surface."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


DEFAULT_BENCHMARK_PATH = workspace_path("evaluation", "event_state_cache_benchmark.json")
DEFAULT_INTEGRATION_PATH = workspace_path("evaluation", "event_state_cache_integration_benchmark.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase18_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase18_completion_gate_summary.txt")
MAX_EVENT_COST = 256
MAX_STATE_BUDGET = 256


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_report(*, benchmark_path: str = DEFAULT_BENCHMARK_PATH, integration_path: str = DEFAULT_INTEGRATION_PATH) -> Dict[str, Any]:
    benchmark = _load_json(benchmark_path)
    integration = _load_json(integration_path)
    metrics = benchmark.get("metrics", {}) if isinstance(benchmark, Mapping) else {}
    profiles = benchmark.get("profiles", {}) if isinstance(benchmark, Mapping) else {}
    integration_metrics = integration.get("metrics", {}) if isinstance(integration, Mapping) else {}
    notes = []
    if isinstance(benchmark, Mapping):
        notes.extend(benchmark.get("policy_notes", []) if isinstance(benchmark.get("policy_notes"), list) else [])
    if isinstance(integration, Mapping):
        notes.extend(integration.get("policy_notes", []) if isinstance(integration.get("policy_notes"), list) else [])
    policy_text = " ".join(str(item).lower() for item in notes)
    required_profiles = {"none", "fixed", "linear", "logarithmic"}
    checks = {
        "benchmark_present": benchmark is not None,
        "benchmark_passed": bool(benchmark and benchmark.get("passed")),
        "benchmark_observed_only": bool(benchmark and benchmark.get("observed_only")),
        "integration_present": integration is not None,
        "integration_passed": bool(integration and integration.get("passed")),
        "integration_observed_only": bool(integration and integration.get("observed_only")),
        "retention_profiles_visible": isinstance(profiles, Mapping) and required_profiles.issubset(set(profiles)),
        "delayed_recall_improves": int(metrics.get("logarithmic_delayed_recall", 0) or 0) > int(metrics.get("fixed_delayed_recall", 0) or 0),
        "logarithmic_state_growth_bounded": float(metrics.get("logarithmic_to_linear_state_ratio", 2.0) or 2.0) < 1.0,
        "negative_abstention_integrity": float(metrics.get("logarithmic_negative_abstention", 0.0) or 0.0) >= 1.0,
        "blocked_decision_integrity": float(metrics.get("blocked_decision_integrity", 0.0) or 0.0) >= 1.0,
        "round_trip_integrity": float(integration_metrics.get("round_trip_integrity", 0.0) or 0.0) >= 1.0,
        "corrupted_state_rejection": float(integration_metrics.get("corrupted_state_rejection", 0.0) or 0.0) >= 1.0,
        "source_revision_integrity": float(integration_metrics.get("source_revision_integrity", 0.0) or 0.0) >= 1.0,
        "reactivation_hint_integrity": float(integration_metrics.get("reactivation_hint_integrity", 0.0) or 0.0) >= 1.0,
        "missing_report_freeze_integrity": float(integration_metrics.get("missing_report_freeze_integrity", 0.0) or 0.0) >= 1.0,
        "source_aware_recall": int(integration_metrics.get("source_aware_logarithmic_delayed_recall", 0) or 0) > int(integration_metrics.get("source_aware_fixed_delayed_recall", 0) or 0),
        "event_cost_bounded": max(int(metrics.get("logarithmic_max_retrieval_event_cost", 0) or 0), int(integration_metrics.get("max_retrieval_event_cost", 0) or 0)) <= MAX_EVENT_COST,
        "state_budget_bounded": int(metrics.get("logarithmic_entry_count", MAX_STATE_BUDGET + 1) or 0) <= MAX_STATE_BUDGET,
        "sparse_verified_policy_visible": all(term in policy_text for term in ("verified", "sparse", "bounded", "dense")),
        "production_memory_unchanged": "does not alter production memory" in policy_text,
    }
    passed = all(checks.values())
    next_actions: List[Dict[str, Any]] = []
    if passed:
        next_actions.append({"priority": 3, "reason": "keep Event Memory observed-only until repeated delayed-recall, abstention, source, energy, and regression review", "command": "python scripts/sara_cli.py eval-operator-dashboard"})
    else:
        next_actions.append({"priority": 1, "reason": "refresh_or_review_phase18_evidence", "command": "python scripts/sara_cli.py eval-research-benchmark-suite"})
    return {
        "schema": "sara-phase18-completion-gate-v1",
        "phase": 18,
        "phase18_complete": passed,
        "status": "phase18_complete" if passed else "phase18_incomplete",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "candidate_count": int(benchmark.get("candidate_count", 0) or 0) if benchmark else 0,
            "logarithmic_delayed_recall": int(metrics.get("logarithmic_delayed_recall", 0) or 0),
            "logarithmic_to_linear_state_ratio": float(metrics.get("logarithmic_to_linear_state_ratio", 0.0) or 0.0),
            "source_aware_logarithmic_delayed_recall": int(integration_metrics.get("source_aware_logarithmic_delayed_recall", 0) or 0),
            "max_retrieval_event_cost": int(integration_metrics.get("max_retrieval_event_cost", 0) or 0),
        },
        "evidence_paths": {"benchmark": benchmark_path, "integration": integration_path},
        "next_actions": next_actions,
        "promotion_rule": {"release_critical": False, "observed_only_until_stable": True, "requires_recall_abstention_source_energy_growth_regression_review": True},
        "what_is_not_proven": [
            "Phase 18 fixture evidence does not prove broad real-world continual-memory generalization.",
            "Proxy retrieval/event cost does not prove physical joule-per-success advantage.",
            "Event Memory is not promoted to release-critical production memory by this gate alone.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 18 verified hierarchical event-state cache evidence.")
    parser.add_argument("--benchmark-path", default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--integration-path", default=DEFAULT_INTEGRATION_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(benchmark_path=args.benchmark_path, integration_path=args.integration_path)
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    lines = ["Phase 18 completion gate", f"status: {report['status']}", f"phase18_complete: {str(report['phase18_complete']).lower()}"]
    lines.extend(f"- {name}: {str(value).lower()}" for name, value in report["checks"].items())
    lines.append("Next actions:")
    lines.extend(f"- {item['reason']} -> {item['command']}" for item in report["next_actions"])
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
