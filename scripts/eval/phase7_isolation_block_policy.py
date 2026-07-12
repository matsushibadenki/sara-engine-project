#!/usr/bin/env python3
"""Apply Phase 7 isolation-audit outcomes to managed collection targets."""

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


DEFAULT_AUDIT_PATH = workspace_path("evaluation", "phase7_isolation_audit.json")
DEFAULT_TARGETS_PATH = workspace_path("autobot", "dataset_builder_collection_targets.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase7_isolation_block_policy.json")
PHASE7_AXES = {
    "metadata_complete",
    "source_hash_isolated",
    "source_revision_isolated",
    "source_domain_isolated",
    "time_split_isolated",
    "near_duplicate_signature_isolated",
}


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_policy(audit: Mapping[str, Any], targets: Mapping[str, Any]) -> Dict[str, Any]:
    checks = audit.get("checks", {}) if isinstance(audit.get("checks"), Mapping) else {}
    target_rows = targets.get("targets", []) if isinstance(targets.get("targets"), list) else []
    request_ids = sorted(
        str(row.get("request_id", "") or "")
        for row in target_rows
        if isinstance(row, Mapping) and str(row.get("request_id", "") or "").startswith("fixture_")
    )
    failed_axes = sorted(axis for axis in PHASE7_AXES if checks.get(axis) is False)
    existing_ids = {
        str(value) for value in targets.get("blocked_request_ids", []) if str(value)
    }
    existing_axes = targets.get("blocked_request_missing_axes", {})
    existing_axes = existing_axes if isinstance(existing_axes, Mapping) else {}
    merged_axes = {
        str(request_id): sorted({str(axis) for axis in axes if str(axis)})
        for request_id, axes in existing_axes.items()
        if str(request_id) and isinstance(axes, list)
    }
    if failed_axes:
        blocked_ids = sorted(existing_ids | set(request_ids))
        for request_id in request_ids:
            merged_axes[request_id] = sorted(set(merged_axes.get(request_id, [])) | set(failed_axes))
        action = "blocked"
    else:
        blocked_ids = sorted(existing_ids - set(request_ids))
        for request_id in request_ids:
            remaining = sorted(set(merged_axes.get(request_id, [])) - PHASE7_AXES)
            if remaining:
                merged_axes[request_id] = remaining
            else:
                merged_axes.pop(request_id, None)
        action = "released"
    updated_targets = dict(targets)
    updated_targets["blocked_request_ids"] = blocked_ids
    updated_targets["blocked_request_missing_axes"] = dict(sorted(merged_axes.items()))
    return {
        "schema": "sara-phase7-isolation-block-policy-v1",
        "action": action,
        "audit_passed": bool(audit.get("passed", False)),
        "failed_axes": failed_axes,
        "affected_request_ids": request_ids,
        "targets": updated_targets,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-path", default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--targets-path", default=DEFAULT_TARGETS_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)
    policy = build_policy(_read_json(args.audit_path), _read_json(args.targets_path))
    targets_path = ensure_parent_directory(args.targets_path)
    with open(targets_path, "w", encoding="utf-8") as handle:
        json.dump(policy["targets"], handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    report = {key: value for key, value in policy.items() if key != "targets"}
    report["targets_path"] = targets_path
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
