#!/usr/bin/env python3
"""Convert a blocked Phase 8 gate into a managed local-reference request."""

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


DEFAULT_GATE_PATH = workspace_path("evaluation", "phase8_completion_gate.json")
DEFAULT_REQUEST_PATH = workspace_path("autobot", "phase8_reference_collection_request.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "phase8_reference_collection_request.json")


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_request(gate: Mapping[str, Any]) -> Dict[str, Any]:
    required_checks = gate.get("required_checks", {})
    required_checks = required_checks if isinstance(required_checks, Mapping) else {}
    missing = sorted(str(name) for name, passed in required_checks.items() if not bool(passed))
    blocked = not bool(gate.get("phase8_complete", False))
    targets = []
    if blocked:
        targets.append(
            {
                "request_id": "phase8_stronger_offline_reference",
                "missing_material_types": ["local_pretrained_embedding_reference"],
                "preferred_reference_types": [
                    "local_pretrained_embedding",
                    "local_pretrained_embedding_faiss",
                    "tiny_cross_encoder",
                ],
                "evaluation_gaps": ["phase8_stronger_ann_reference"],
                "missing_checks": missing,
                "requirements": {
                    "same_corpus": True,
                    "same_query_set": True,
                    "same_success_criteria": True,
                    "cpu_only": True,
                    "local_files_only": True,
                    "record_model_identity": True,
                    "record_embedding_dimension": True,
                    "record_latency_memory_and_quality": True,
                },
                "rerun_command": "python scripts/sara_cli.py eval-phase8-evidence-cycle --pretrained-embedding-model <managed-model-directory>",
            }
        )
    return {
        "schema": "sara-phase8-reference-collection-request-v1",
        "source_gate_schema": str(gate.get("schema", "")),
        "blocked": blocked,
        "gate_status": str(gate.get("status", "")),
        "target_count": len(targets),
        "targets": targets,
        "policy_notes": [
            "This request does not download models or claim Phase 8 completion.",
            "A local reference must be evaluated on the same corpus, query set, success criteria, and CPU boundary.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-path", default=DEFAULT_GATE_PATH)
    parser.add_argument("--request-path", default=DEFAULT_REQUEST_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)
    request = build_request(_read_json(args.gate_path))
    with open(ensure_parent_directory(args.request_path), "w", encoding="utf-8") as handle:
        json.dump(request, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    report = dict(request)
    report["request_path"] = args.request_path
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
