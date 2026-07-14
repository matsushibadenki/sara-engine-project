#!/usr/bin/env python3
"""Validate the managed Phase 12 operator-experience surface."""

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

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402

DEFAULT_DASHBOARD_PATH = workspace_path("evaluation", "operator_dashboard.json")
DEFAULT_GUIDE_PATH = os.path.join(PROJECT_ROOT, "doc", "OPERATOR_GUIDE.md")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase12_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase12_completion_gate_summary.txt")
REQUIRED_ARTIFACTS = {"phase6", "phase7", "phase8", "phase9", "phase10", "phase11", "phase13", "phase14", "phase15", "phase16", "phase17", "phase18", "phase19", "phase20", "research_product", "release"}


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _check_dashboard(path: str) -> Dict[str, Any]:
    report = _load_json(path)
    errors: List[str] = []
    if report is None:
        return {"passed": False, "errors": [f"Missing or invalid operator dashboard: {path}"]}
    if report.get("schema") != "sara-operator-dashboard-v1":
        errors.append("Operator dashboard schema is invalid.")
    states = report.get("artifact_states", {}) if isinstance(report.get("artifact_states"), Mapping) else {}
    missing = sorted(REQUIRED_ARTIFACTS - set(states))
    if missing:
        errors.append("Dashboard is missing artifact states: " + ", ".join(missing))
    actions = report.get("next_actions", [])
    if not isinstance(actions, list) or not actions:
        errors.append("Dashboard has no actionable next step.")
    commands = report.get("operator_commands", {})
    if not isinstance(commands, Mapping) or not commands.get("refresh_dashboard"):
        errors.append("Dashboard is missing its refresh command.")
    if not report.get("what_is_proven") or not report.get("what_is_not_proven"):
        errors.append("Dashboard must show both proven and not-proven evidence.")
    return {"passed": not errors, "errors": errors, "artifact_count": len(states), "action_count": len(actions) if isinstance(actions, list) else 0}


def _check_guide(path: str) -> Dict[str, Any]:
    try:
        text = open(path, "r", encoding="utf-8").read()
    except OSError:
        return {"passed": False, "errors": [f"Missing operator guide: {path}"]}
    required = ["Daily Review", "Reproduce Evidence", "Troubleshooting", "Managed output violation", "Physical energy pending"]
    missing = [item for item in required if item not in text]
    return {"passed": not missing, "errors": ["Operator guide is missing: " + ", ".join(missing)] if missing else [], "section_count": len(required)}


def build_report(*, dashboard_path: str, guide_path: str) -> Dict[str, Any]:
    checks = {"operator_dashboard": _check_dashboard(dashboard_path), "operator_guide": _check_guide(guide_path)}
    passed = all(bool(item.get("passed")) for item in checks.values())
    return {
        "schema": "sara-phase12-completion-gate-v1",
        "phase": 12,
        "phase12_complete": passed,
        "status": "phase12_complete" if passed else "phase12_incomplete",
        "passed": passed,
        "checks": checks,
        "policy": {"cpu_first": True, "managed_outputs_only": True, "optional_integrations_non_blocking": True},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 12 operator experience evidence.")
    parser.add_argument("--dashboard-path", default=DEFAULT_DASHBOARD_PATH)
    parser.add_argument("--guide-path", default=DEFAULT_GUIDE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(dashboard_path=args.dashboard_path, guide_path=args.guide_path)
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    lines = ["Phase 12 completion gate", f"status: {report['status']}", f"phase12_complete: {str(report['phase12_complete']).lower()}"]
    for name, check in report["checks"].items():
        lines.append(f"{name}: {str(check['passed']).lower()}")
        lines.extend(f"error: {error}" for error in check.get("errors", []))
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
