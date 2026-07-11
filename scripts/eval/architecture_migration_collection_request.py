#!/usr/bin/env python3
"""Convert blocked architecture-migration provenance gates into collection targets."""

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


DEFAULT_GATE_PATH = workspace_path("evaluation", "architecture_migration_external_gate.json")
DEFAULT_TARGETS_PATH = workspace_path("autobot", "architecture_migration_collection_targets.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "architecture_migration_collection_request.json")


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_targets(gate: Mapping[str, Any]) -> Dict[str, Any]:
    blocked_reasons = [str(item) for item in gate.get("blocked_reasons", ()) if str(item)]
    targets = []
    if blocked_reasons:
        targets.append(
            {
                "request_id": "architecture_migration_external_provenance",
                "missing_material_types": ["transcript_segment", "revision_note"],
                "preferred_material_types": ["source_claim", "qa_pair", "transcript_segment"],
                "evaluation_gaps": ["architecture_migration_external_provenance"],
                "candidate_source_domains": [],
                "architecture_migration_requirements": {
                    "minimum_distinct_https_source_sites": 2,
                    "exclude_domains": ["example.org"],
                    "require_observed_only": True,
                    "require_compliance_level": "allow",
                    "require_unique_material_hashes": True,
                    "minimum_records_per_independent_source_site": 3,
                },
                "blocked_reasons": blocked_reasons,
            }
        )
    return {
        "schema": "sara-architecture-migration-collection-targets-v1",
        "source_gate_schema": str(gate.get("schema", "")),
        "target_count": len(targets),
        "targets": targets,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-path", default=DEFAULT_GATE_PATH)
    parser.add_argument("--targets-path", default=DEFAULT_TARGETS_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    targets = build_targets(_read_json(args.gate_path))
    targets_path = ensure_parent_directory(args.targets_path)
    with open(targets_path, "w", encoding="utf-8") as handle:
        json.dump(targets, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    report = {"schema": "sara-architecture-migration-collection-request-v1", "targets_path": targets_path, **targets}
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
