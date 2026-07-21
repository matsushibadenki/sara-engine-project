#!/usr/bin/env python3
"""Turn a blocked Phase 22 external gate into managed collection targets."""

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


DEFAULT_GATE = workspace_path("evaluation", "continual_horizon_external_gate.json")
DEFAULT_TARGETS = workspace_path("autobot", "continual_horizon_collection_targets.json")
DEFAULT_REPORT = workspace_path("evaluation", "continual_horizon_collection_request.json")


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_targets(gate: Mapping[str, Any]) -> Dict[str, Any]:
    promotion_checks = gate.get("promotion_checks", {})
    blocked = not bool(gate.get("promotion_allowed", False))
    domains = [str(item) for item in gate.get("source_domains", []) if str(item)]
    target_rows = []
    if blocked:
        for domain in domains or ["new_independent_domain"]:
            target_rows.append(
                {
                    "target_id": f"phase22:{domain}",
                    "source_domain": domain,
                    "required_horizon_buckets": [10, 30, 100],
                    "minimum_records_per_bucket": 1,
                    "required_fields": [
                        "source_ref",
                        "source_hash",
                        "source_revision",
                        "source_domain",
                        "collection_time",
                        "near_duplicate_signature",
                        "evidence_scope",
                        "observed_only",
                        "compliance_level",
                    ],
                    "deduplication": {
                        "unique_source_hash": True,
                        "unique_source_ref": True,
                        "no_shared_revision_across_eval_boundaries": True,
                    },
                    "quality_constraints": {
                        "evidence_scope": "independent_external",
                        "observed_only": True,
                        "compliance_level": "allow",
                    },
                    "collection_policy": "collect_only; do_not_fabricate; do_not_promote_until_gate_passes",
                }
            )
    return {
        "schema": "sara-continual-horizon-collection-targets-v1",
        "source_gate_schema": str(gate.get("schema", "")),
        "promotion_allowed_at_generation": bool(gate.get("promotion_allowed", False)),
        "blocked_promotion_checks": [
            key for key, value in promotion_checks.items() if not bool(value)
        ],
        "target_count": len(target_rows),
        "targets": target_rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-path", default=DEFAULT_GATE)
    parser.add_argument("--targets-path", default=DEFAULT_TARGETS)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    targets = build_targets(_read_json(args.gate_path))
    targets_path = ensure_parent_directory(args.targets_path)
    with open(targets_path, "w", encoding="utf-8") as handle:
        json.dump(targets, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    report = {
        "schema": "sara-continual-horizon-collection-request-v1",
        "targets_path": targets_path,
        **targets,
    }
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
