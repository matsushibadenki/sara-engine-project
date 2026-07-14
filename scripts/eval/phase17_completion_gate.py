#!/usr/bin/env python3
"""Validate the observed-only Phase 17 verified sparse resonance credit surface."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


DEFAULT_CREDIT_PATH = workspace_path("evaluation", "resonance_credit_benchmark.json")
DEFAULT_INTEGRATION_PATH = workspace_path("evaluation", "resonance_credit_integration_benchmark.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase17_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase17_completion_gate_summary.txt")
MAX_EVENT_COST = 256
MAX_STATE_BUDGET = 256


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_report(*, credit_path: str = DEFAULT_CREDIT_PATH, integration_path: str = DEFAULT_INTEGRATION_PATH) -> Dict[str, Any]:
    credit = _load_json(credit_path)
    integration = _load_json(integration_path)
    credit_metrics = credit.get("metrics", {}) if isinstance(credit, Mapping) else {}
    integration_metrics = integration.get("metrics", {}) if isinstance(integration, Mapping) else {}
    credit_notes = credit.get("policy_notes", []) if isinstance(credit, Mapping) else []
    integration_notes = integration.get("policy_notes", []) if isinstance(integration, Mapping) else []
    policy_text = " ".join(str(item).lower() for item in list(credit_notes) + list(integration_notes))
    credit_rows = credit.get("rows", []) if isinstance(credit, Mapping) else []
    integration_rows = integration.get("rows", []) if isinstance(integration, Mapping) else []
    credit_rows = credit_rows if isinstance(credit_rows, list) else []
    integration_rows = integration_rows if isinstance(integration_rows, list) else []
    freeze_decisions = {"freeze_contradiction", "freeze_abstention", "freeze_source", "freeze_metabolic_budget", "freeze_resonance"}
    checks = {
        "credit_present": credit is not None,
        "credit_passed": bool(credit and credit.get("passed")),
        "credit_observed_only": bool(credit and credit.get("observed_only")),
        "integration_present": integration is not None,
        "integration_passed": bool(integration and integration.get("passed")),
        "integration_observed_only": bool(integration and integration.get("observed_only")),
        "decision_integrity": float(credit_metrics.get("decision_integrity", 0.0) or 0.0) >= 1.0 and float(integration_metrics.get("decision_integrity", 0.0) or 0.0) >= 1.0,
        "harmful_update_suppression": float(credit_metrics.get("harmful_update_suppression", 0.0) or 0.0) >= 1.0 and int(credit_metrics.get("naive_reward_harmful_update_count", 0) or 0) > 0,
        "freeze_reason_coverage": any(str(row.get("decision", "")) in freeze_decisions for row in credit_rows if isinstance(row, Mapping)),
        "multi_signal_updates_visible": int(credit_metrics.get("resonance_update_count", 0) or 0) > 0,
        "source_backed_integration": float(integration_metrics.get("live_source_backed_integrity", 0.0) or 0.0) >= 1.0 and len(integration.get("source_paths", {}) if isinstance(integration, Mapping) and isinstance(integration.get("source_paths"), Mapping) else (integration.get("source_paths", []) if isinstance(integration, Mapping) and isinstance(integration.get("source_paths"), list) else [])) >= 5,
        "event_cost_bounded": max(int(credit_metrics.get("max_event_cost", 0) or 0), int(integration_metrics.get("max_combined_event_cost", 0) or 0)) <= MAX_EVENT_COST,
        "state_budget_bounded": int(credit_metrics.get("max_state_budget_units", MAX_STATE_BUDGET + 1) or 0) <= MAX_STATE_BUDGET,
        "sparse_local_policy_visible": all(term in policy_text for term in ("sparse", "local", "cpu-first", "observed-only")),
        "production_learning_unchanged": "does not alter production learning" in policy_text,
    }
    passed = all(checks.values())
    next_actions: List[Dict[str, Any]] = []
    if passed:
        next_actions.append({"priority": 3, "reason": "keep resonance credit observed-only until repeated quality, freeze, energy, and regression review", "command": "python scripts/sara_cli.py eval-operator-dashboard"})
    else:
        next_actions.append({"priority": 1, "reason": "refresh_or_review_phase17_evidence", "command": "python scripts/sara_cli.py eval-research-benchmark-suite"})
    return {
        "schema": "sara-phase17-completion-gate-v1",
        "phase": 17,
        "phase17_complete": passed,
        "status": "phase17_complete" if passed else "phase17_incomplete",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "credit_case_count": int(credit.get("case_count", 0) or 0) if credit else 0,
            "integration_case_count": int(integration.get("case_count", 0) or 0) if integration else 0,
            "resonance_update_count": int(credit_metrics.get("resonance_update_count", 0) or 0),
            "resonance_freeze_count": int(credit_metrics.get("resonance_freeze_count", 0) or 0),
            "max_combined_event_cost": int(integration_metrics.get("max_combined_event_cost", 0) or 0),
        },
        "evidence_paths": {"credit": credit_path, "integration": integration_path},
        "next_actions": next_actions,
        "promotion_rule": {"release_critical": False, "observed_only_until_stable": True, "requires_quality_freeze_energy_trace_regression_review": True},
        "what_is_not_proven": [
            "Phase 17 fixture evidence does not prove broad external generalization or biological equivalence.",
            "Proxy event cost does not prove physical joule-per-success advantage.",
            "Verified resonance credit is not promoted to release-critical production learning by this gate alone.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 17 verified sparse resonance credit evidence.")
    parser.add_argument("--credit-path", default=DEFAULT_CREDIT_PATH)
    parser.add_argument("--integration-path", default=DEFAULT_INTEGRATION_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(credit_path=args.credit_path, integration_path=args.integration_path)
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    lines = ["Phase 17 completion gate", f"status: {report['status']}", f"phase17_complete: {str(report['phase17_complete']).lower()}"]
    lines.extend(f"- {name}: {str(value).lower()}" for name, value in report["checks"].items())
    lines.append("Next actions:")
    lines.extend(f"- {item['reason']} -> {item['command']}" for item in report["next_actions"])
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
