#!/usr/bin/env python3
"""Build managed collection targets for a blocked Phase 23 external gate."""

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

DEFAULT_GATE = workspace_path("evaluation", "phase23_external_multimodal_gate.json")
DEFAULT_TARGETS = workspace_path("autobot", "phase23_multimodal_collection_targets.json")
DEFAULT_REPORT = workspace_path("evaluation", "phase23_multimodal_collection_request.json")

CASE_TARGETS = (
    ("verified_aligned", "verify_cross_modal_structure", 2, ["vision", "audio"]),
    ("missing_modality", "provisional_missing_modality_prediction", 1, ["vision"]),
    ("contradictory", "abstain_cross_modal_contradiction", 1, ["vision", "audio"]),
    ("temporal_misalignment", "abstain_temporal_misalignment", 1, ["vision", "audio"]),
)


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_targets(gate: Mapping[str, Any]) -> Dict[str, Any]:
    blocked = not bool(gate.get("promotion_allowed", False))
    targets = []
    if blocked:
        for case_type, decision, minimum_count, modalities in CASE_TARGETS:
            targets.append(
                {
                    "target_id": f"phase23:{case_type}",
                    "case_type": case_type,
                    "expected_decision": decision,
                    "minimum_case_count": minimum_count,
                    "required_modalities": modalities,
                    "required_fields": [
                        "case_id",
                        "source_ref",
                        "source_hash",
                        "source_revision",
                        "source_domain",
                        "collection_time",
                        "license_hint",
                        "near_duplicate_signature",
                        "evidence_scope",
                        "observed_only",
                        "compliance_level",
                        "expected_modalities",
                        "evidence[].modality",
                        "evidence[].label",
                        "evidence[].claim_key",
                        "evidence[].timestamp_ms",
                        "evidence[].source_ref",
                        "evidence[].source_hash",
                    ],
                    "quality_constraints": {
                        "evidence_scope": "independent_external",
                        "observed_only": True,
                        "compliance_level": "allow",
                        "fixture_or_generated_source": False,
                        "minimum_independent_domains_across_batch": 2,
                    },
                    "collection_policy": (
                        "collect_only; preserve rights and timestamps; "
                        "do_not_fabricate; do_not_promote_until_gate_passes"
                    ),
                }
            )
    return {
        "schema": "sara-phase23-multimodal-collection-targets-v1",
        "source_gate_schema": str(gate.get("schema", "")),
        "promotion_allowed_at_generation": bool(gate.get("promotion_allowed", False)),
        "target_count": len(targets),
        "targets": targets,
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
        "schema": "sara-phase23-multimodal-collection-request-v1",
        "targets_path": targets_path,
        **targets,
    }
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
