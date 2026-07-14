#!/usr/bin/env python3
"""Validate the managed Phase 11 neuromorphic portability evidence."""

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


DEFAULT_MATRIX_PATH = workspace_path("evaluation", "neuromorphic_capability_matrix.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase11_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase11_completion_gate_summary.txt")
REQUIRED_PROFILES = {"lava", "spinnaker", "akida"}


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _check_matrix(path: str) -> Dict[str, Any]:
    report = _load_json(path)
    errors: List[str] = []
    if report is None:
        return {"passed": False, "errors": [f"Missing or invalid capability matrix: {path}"]}
    if report.get("schema") != "sara-neuromorphic-capability-matrix-report-v1":
        errors.append("Capability matrix report schema is invalid.")
    if not bool(report.get("passed")):
        errors.append("Capability matrix generation did not pass.")
    profiles = set(str(item) for item in report.get("profiles", []) if str(item))
    missing_profiles = sorted(REQUIRED_PROFILES - profiles)
    if missing_profiles:
        errors.append("Required backend profiles are missing: " + ", ".join(missing_profiles))
    matrix = report.get("capability_matrix", {})
    matrix = matrix if isinstance(matrix, Mapping) else {}
    if not bool(matrix.get("all_profiles_compatible")):
        errors.append("Not all configured neuromorphic profiles are compatible.")
    common_ir = matrix.get("common_event_ir", {})
    common_ir = common_ir if isinstance(common_ir, Mapping) else {}
    if common_ir.get("schema") != "sara-spike-event-ir-v1":
        errors.append("Common event IR schema is missing or invalid.")
    if not bool(common_ir.get("budget_ok")) or int(common_ir.get("event_count", 0) or 0) <= 0:
        errors.append("Common event IR is empty or exceeds its state/event budget.")
    profile_rows = matrix.get("profiles", {})
    profile_rows = profile_rows if isinstance(profile_rows, Mapping) else {}
    missing_fallback_fields = []
    for profile in sorted(REQUIRED_PROFILES):
        row = profile_rows.get(profile, {})
        if not isinstance(row, Mapping):
            missing_fallback_fields.append(profile)
            continue
        for field in ("adapter", "state_trace_adapter_policy", "unsupported_operations", "notes"):
            if field not in row:
                missing_fallback_fields.append(f"{profile}.{field}")
    if missing_fallback_fields:
        errors.append("Profile fallback/unsupported fields are missing: " + ", ".join(missing_fallback_fields))
    cpu_reference = report.get("cpu_reference", {})
    cpu_reference = cpu_reference if isinstance(cpu_reference, Mapping) else {}
    if not bool(cpu_reference.get("validated")) or not bool(cpu_reference.get("release_critical")):
        errors.append("CPU reference is not marked validated and release-critical.")
    if bool(report.get("hardware_runtime_required", True)):
        errors.append("Hardware runtime is incorrectly marked as required.")
    return {
        "passed": not errors,
        "errors": errors,
        "profile_count": len(profiles),
        "profiles": sorted(profiles),
        "event_count": int(common_ir.get("event_count", 0) or 0),
        "unsupported_summary": matrix.get("unsupported_summary", {}),
        "matrix_path": os.path.abspath(path),
    }


def build_report(*, matrix_path: str) -> Dict[str, Any]:
    checks = {"neuromorphic_portability_matrix": _check_matrix(matrix_path)}
    passed = all(bool(item.get("passed")) for item in checks.values())
    return {
        "schema": "sara-phase11-completion-gate-v1",
        "phase": 11,
        "phase11_complete": passed,
        "status": "phase11_complete" if passed else "phase11_incomplete",
        "passed": passed,
        "checks": checks,
        "policy": {
            "cpu_reference_required": True,
            "hardware_specific_adapters_optional": True,
            "accelerator_runtime_required": False,
            "unsupported_operations_must_be_visible": True,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 11 neuromorphic portability evidence.")
    parser.add_argument("--matrix-path", default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(matrix_path=args.matrix_path)
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    check = report["checks"]["neuromorphic_portability_matrix"]
    lines = [
        "Phase 11 completion gate",
        f"status: {report['status']}",
        f"phase11_complete: {str(report['phase11_complete']).lower()}",
        f"profiles: {','.join(check.get('profiles', [])) or 'none'}",
        f"event_count: {check.get('event_count', 0)}",
    ]
    for error in check.get("errors", []):
        lines.append(f"error: {error}")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
