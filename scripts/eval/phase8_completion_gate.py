#!/usr/bin/env python3
"""Classify Phase 8 baseline implementation readiness and evidence completion."""

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


DEFAULT_COMPARISON_PATH = workspace_path("evaluation", "sara_ann_comparison_report.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase8_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase8_completion_gate_summary.txt")


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_report(comparison: Mapping[str, Any]) -> Dict[str, Any]:
    checks = comparison.get("checks", {}) if isinstance(comparison.get("checks"), Mapping) else {}
    implementation_ready = bool(
        comparison.get("schema") == "sara-ann-comparison-report-v1"
        and checks
        and isinstance(comparison.get("baseline_cards"), list)
        and isinstance(comparison.get("reference_readiness"), Mapping)
    )
    required_checks = (
        "external_validity_passed",
        "ladder_passed",
        "bm25_reference_present",
        "stronger_real_reference_present",
        "per_task_summary_present",
        "quality_and_cost_reported_together",
        "offline_references_labeled",
    )
    phase8_evidence_complete = bool(
        implementation_ready and all(bool(checks.get(name, False)) for name in required_checks)
    )
    if phase8_evidence_complete:
        status = "phase8_complete"
        next_action = "Use the frozen offline-reference comparison in downstream research and release reports."
    elif implementation_ready:
        status = "implementation_complete_stronger_baseline_pending"
        next_action = "Configure and freeze at least one local pretrained embedding or tiny-Transformer reference on the same held-out workload."
    else:
        status = "implementation_repair_required"
        next_action = "Regenerate the SARA-versus-ANN comparison artifact with reference readiness and labeled baseline cards."
    return {
        "schema": "sara-phase8-completion-gate-v1",
        "status": status,
        "implementation_ready": implementation_ready,
        "phase8_evidence_complete": phase8_evidence_complete,
        "phase8_complete": phase8_evidence_complete,
        "next_action": next_action,
        "required_checks": {name: bool(checks.get(name, False)) for name in required_checks},
        "comparison_status": str(comparison.get("status", "")),
        "physical_evidence_separate": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-path", default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(_load_json(args.comparison_path))
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(ensure_parent_directory(args.summary_path), "w", encoding="utf-8") as handle:
        handle.write(f"Phase 8 completion gate: {report['status']}\n")
        handle.write(f"Implementation ready: {report['implementation_ready']}\n")
        handle.write(f"Baseline evidence complete: {report['phase8_evidence_complete']}\n")
        handle.write(f"Next action: {report['next_action']}\n")
    return 0 if report["phase8_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
