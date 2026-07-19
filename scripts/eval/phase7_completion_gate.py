#!/usr/bin/env python3
"""Classify Phase 7 implementation readiness separately from isolation evidence."""

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


DEFAULT_READINESS_PATH = workspace_path("evaluation", "autobot_gap_loop_readiness.json")
DEFAULT_ISOLATION_PATH = workspace_path("evaluation", "phase7_isolation_audit.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase7_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase7_completion_gate_summary.txt")


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_report(readiness: Mapping[str, Any], isolation: Mapping[str, Any]) -> Dict[str, Any]:
    readiness_checks = readiness.get("checks", {}) if isinstance(readiness.get("checks"), Mapping) else {}
    isolation_checks = isolation.get("checks", {}) if isinstance(isolation.get("checks"), Mapping) else {}
    readiness_schema_valid = readiness.get("schema") == "sara-autobot-gap-loop-readiness-v1"
    isolation_schema_valid = isolation.get("schema") == "sara-phase7-isolation-audit-v1"
    implementation_ready = bool(readiness_schema_valid and readiness_checks and isolation_schema_valid and isolation_checks)
    gap_loop_ready = bool(readiness.get("passed", False))
    isolation_evidence_complete = bool(
        isolation.get("passed", False)
        and isolation_checks.get("independent_evidence_scope_valid", False)
    )
    phase7_complete = bool(implementation_ready and gap_loop_ready and isolation_evidence_complete)
    if phase7_complete:
        status = "phase7_complete"
        next_action = "Keep the isolated material split pinned; rerun downstream evidence only when the split or source data changes."
    elif implementation_ready and not isolation_evidence_complete:
        status = "implementation_complete_isolation_evidence_pending"
        next_action = "Freeze independent train and evaluation material manifests, then resolve every reported provenance or overlap check."
    elif implementation_ready:
        status = "gap_loop_repair_required"
        next_action = "Repair the managed collection, gap-material, or curriculum queue artifacts before rerunning the gate."
    else:
        status = "implementation_repair_required"
        next_action = "Generate valid Phase 7 readiness and isolation-audit artifacts before promoting autonomous materials."
    metrics = isolation.get("metrics", {}) if isinstance(isolation.get("metrics"), Mapping) else {}
    return {
        "schema": "sara-phase7-completion-gate-v1",
        "status": status,
        "implementation_ready": implementation_ready,
        "gap_loop_ready": gap_loop_ready,
        "isolation_evidence_complete": isolation_evidence_complete,
        "phase7_complete": phase7_complete,
        "next_action": next_action,
        "readiness_passed": bool(readiness.get("passed", False)),
        "independent_evidence_scope_valid": bool(
            isolation_checks.get("independent_evidence_scope_valid", False)
        ),
        "isolation_checks": dict(isolation_checks),
        "isolation_metrics": {
            "train_row_count": int(metrics.get("train_row_count", 0) or 0),
            "evaluation_row_count": int(metrics.get("evaluation_row_count", 0) or 0),
            "shared_source_hash_count": len(metrics.get("shared_source_hashes", []) if isinstance(metrics.get("shared_source_hashes"), list) else []),
            "shared_source_revision_count": len(metrics.get("shared_source_revisions", []) if isinstance(metrics.get("shared_source_revisions"), list) else []),
            "shared_source_domain_count": len(metrics.get("shared_source_domains", []) if isinstance(metrics.get("shared_source_domains"), list) else []),
            "near_duplicate_pair_count": len(metrics.get("near_duplicate_pairs", []) if isinstance(metrics.get("near_duplicate_pairs"), list) else []),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-path", default=DEFAULT_READINESS_PATH)
    parser.add_argument("--isolation-path", default=DEFAULT_ISOLATION_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(_load_json(args.readiness_path), _load_json(args.isolation_path))
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(ensure_parent_directory(args.summary_path), "w", encoding="utf-8") as handle:
        handle.write(f"Phase 7 completion gate: {report['status']}\n")
        handle.write(f"Implementation ready: {report['implementation_ready']}\n")
        handle.write(f"Gap loop ready: {report['gap_loop_ready']}\n")
        handle.write(f"Isolation evidence complete: {report['isolation_evidence_complete']}\n")
        handle.write(f"Next action: {report['next_action']}\n")
    return 0 if report["phase7_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
