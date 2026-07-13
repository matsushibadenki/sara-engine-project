#!/usr/bin/env python3
"""Classify Phase 6 implementation readiness separately from physical evidence completion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_READINESS_PATH = workspace_path("evaluation", "energy_measurement_readiness.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase6_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase6_completion_gate_summary.txt")


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_report(readiness: Mapping[str, Any]) -> Dict[str, Any]:
    checks = readiness.get("checks", {}) if isinstance(readiness.get("checks"), Mapping) else {}
    progress = readiness.get("measurement_session_progress", {})
    progress = progress if isinstance(progress, Mapping) else {}
    measurement_plan = readiness.get("measurement_plan", {})
    measurement_plan = measurement_plan if isinstance(measurement_plan, Mapping) else {}
    session_plan = readiness.get("measurement_session_plan", {})
    session_plan = session_plan if isinstance(session_plan, Mapping) else {}
    protocol_ready = bool(
        readiness.get("protocol_ready", False)
        or readiness.get("status") in {
            "protocol_ready_pending_measurements",
            "real_joule_evidence_passed",
        }
        or (
            readiness.get("schema") == "sara-energy-measurement-readiness-v2"
            and bool(checks.get("schema_ready", False))
        )
    )
    session_plan_available = bool(readiness.get("measurement_session_plan"))
    implementation_ready = bool(protocol_ready and session_plan_available and checks)
    real_rows_present = bool(
        readiness.get("has_real_measurements", False)
        or readiness.get("real_joule_measurements_present", False)
    )
    physical_evidence_complete = bool(
        readiness.get("status") == "real_joule_evidence_passed"
        and readiness.get("passed", False)
        and progress.get("status") == "complete"
    )
    if physical_evidence_complete:
        status = "phase6_complete"
        next_action = "Regenerate downstream Phase 6, Phase 8, research-product, and release reports."
    elif implementation_ready and not real_rows_present:
        status = "implementation_complete_physical_measurement_pending"
        next_action = "Execute the frozen physical-energy session and append matched SARA/ANN joule rows."
    elif implementation_ready:
        status = "physical_measurement_repair_required"
        next_action = "Resolve invalid, partial, or failing paired measurements before rerunning the gate."
    else:
        status = "implementation_repair_required"
        next_action = "Regenerate the Phase 6 readiness artifact and repair missing protocol or session-plan fields."
    return {
        "schema": "sara-phase6-completion-gate-v1",
        "status": status,
        "implementation_ready": implementation_ready,
        "physical_evidence_complete": physical_evidence_complete,
        "phase6_complete": physical_evidence_complete,
        "next_action": next_action,
        "readiness_status": str(readiness.get("status", "")),
        "real_measurement_row_count": int(readiness.get("measurement_count", 0) or 0),
        "session_progress": {
            "status": str(progress.get("status", "")),
            "planned_pair_count": int(
                progress.get(
                    "planned_pair_count",
                    measurement_plan.get(
                        "pending_pair_count",
                        session_plan.get("planned_pair_count", 0),
                    ),
                )
                or 0
            ),
            "complete_valid_pair_count": int(progress.get("complete_valid_pair_count", 0) or 0),
            "invalid_pair_count": int(progress.get("invalid_pair_count", 0) or 0),
            "partial_pair_count": int(progress.get("partial_pair_count", 0) or 0),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-path", default=DEFAULT_READINESS_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(_load_json(args.readiness_path))
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(ensure_parent_directory(args.summary_path), "w", encoding="utf-8") as handle:
        handle.write(f"Phase 6 completion gate: {report['status']}\n")
        handle.write(f"Implementation ready: {report['implementation_ready']}\n")
        handle.write(f"Physical evidence complete: {report['physical_evidence_complete']}\n")
        handle.write(f"Next action: {report['next_action']}\n")
    return 0 if report["phase6_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
