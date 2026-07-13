#!/usr/bin/env python3
"""Validate the managed Phase 9 research benchmark package."""

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


DEFAULT_MANIFEST_PATH = workspace_path("evaluation", "research_benchmark_manifest.json")
DEFAULT_PROTOCOL_PATH = os.path.join(PROJECT_ROOT, "doc", "BENCHMARK_PROTOCOL.md")
DEFAULT_FIXTURE_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "benchmark_fixtures", "external_validity_cases.jsonl"
)
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase9_completion_gate.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "phase9_completion_gate_summary.txt")


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _check_manifest(path: str) -> Dict[str, Any]:
    manifest = _load_json(path)
    errors: List[str] = []
    if manifest is None:
        return {"passed": False, "errors": [f"Missing or invalid benchmark manifest: {path}"]}
    if manifest.get("schema") != "sara-research-benchmark-manifest-v1":
        errors.append("Benchmark manifest schema is not sara-research-benchmark-manifest-v1.")
    if bool(manifest.get("dry_run")):
        errors.append("Phase 9 completion cannot promote a dry-run manifest.")
    commands = manifest.get("commands", [])
    if not isinstance(commands, list) or not commands:
        errors.append("Benchmark manifest has no executed commands.")
        commands = []
    failed_commands = []
    unmanaged_outputs = []
    missing_outputs = []
    for item in commands:
        if not isinstance(item, Mapping) or item.get("status") != "passed" or item.get("returncode") != 0:
            failed_commands.append(item.get("command_id", "unknown") if isinstance(item, Mapping) else "invalid")
        for output in item.get("managed_outputs", []) if isinstance(item, Mapping) else []:
            output_path = os.path.abspath(str(output))
            allowed = any(
                output_path == root or output_path.startswith(root + os.sep)
                for root in (
                    os.path.join(PROJECT_ROOT, "data"),
                    os.path.join(PROJECT_ROOT, "workspace"),
                    os.path.join(PROJECT_ROOT, "models"),
                )
            )
            if not allowed:
                unmanaged_outputs.append(output)
            if not os.path.exists(output_path):
                missing_outputs.append(output)
    if failed_commands:
        errors.append("Benchmark commands did not all pass: " + ", ".join(map(str, failed_commands)))
    if unmanaged_outputs:
        errors.append("Benchmark declares unmanaged outputs: " + ", ".join(map(str, unmanaged_outputs)))
    if missing_outputs:
        errors.append("Benchmark outputs are missing: " + ", ".join(map(str, missing_outputs)))
    if not isinstance(manifest.get("what_is_proven"), list) or not manifest["what_is_proven"]:
        errors.append("Manifest is missing a non-empty what_is_proven section.")
    if not isinstance(manifest.get("what_is_not_proven"), list) or not manifest["what_is_not_proven"]:
        errors.append("Manifest is missing a non-empty what_is_not_proven section.")
    return {
        "passed": not errors,
        "errors": errors,
        "command_count": len(commands),
        "failed_command_count": len(failed_commands),
        "missing_output_count": len(missing_outputs),
        "unmanaged_output_count": len(unmanaged_outputs),
        "manifest_path": os.path.abspath(path),
    }


def _check_protocol(path: str) -> Dict[str, Any]:
    try:
        text = open(path, "r", encoding="utf-8").read()
    except OSError:
        return {"passed": False, "errors": [f"Missing benchmark protocol: {path}"]}
    required = ["Recommended Command", "What Is Proven", "What Is Not Proven", "Output Policy"]
    missing = [heading for heading in required if heading not in text]
    return {
        "passed": not missing,
        "errors": ["Benchmark protocol is missing: " + ", ".join(missing)] if missing else [],
        "required_section_count": len(required),
        "protocol_path": os.path.abspath(path),
    }


def _check_fixture(path: str) -> Dict[str, Any]:
    required_types = {"qa", "negative", "partial", "contrastive", "noisy", "adversarial", "delayed"}
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"Invalid benchmark fixture: {exc}"]}
    observed = {str(row.get("task_type", "")) for row in rows}
    missing = sorted(required_types - observed)
    return {
        "passed": bool(rows) and not missing,
        "errors": ["Benchmark fixture is missing case types: " + ", ".join(missing)] if missing else [],
        "case_count": len(rows),
        "case_types": sorted(observed),
        "fixture_path": os.path.abspath(path),
    }


def build_report(*, manifest_path: str, protocol_path: str, fixture_path: str) -> Dict[str, Any]:
    checks = {
        "executed_suite": _check_manifest(manifest_path),
        "benchmark_protocol": _check_protocol(protocol_path),
        "repository_safe_fixture": _check_fixture(fixture_path),
    }
    passed = all(bool(check.get("passed")) for check in checks.values())
    return {
        "schema": "sara-phase9-completion-gate-v1",
        "phase": 9,
        "phase9_complete": passed,
        "status": "phase9_complete" if passed else "phase9_incomplete",
        "passed": passed,
        "checks": checks,
        "claim_policy": {
            "physical_energy_is_optional": True,
            "unproven_claims_must_remain_labeled": True,
            "managed_outputs_only": True,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the managed Phase 9 benchmark package.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--protocol-path", default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(
        manifest_path=args.manifest_path,
        protocol_path=args.protocol_path,
        fixture_path=args.fixture_path,
    )
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    lines = [
        "Phase 9 completion gate",
        f"status: {report['status']}",
        f"phase9_complete: {str(report['phase9_complete']).lower()}",
    ]
    for name, check in report["checks"].items():
        lines.append(f"{name}: {str(check['passed']).lower()}")
        for error in check.get("errors", []):
            lines.append(f"error: {error}")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
