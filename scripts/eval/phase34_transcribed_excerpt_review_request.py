#!/usr/bin/env python3
"""Build the evidence-bound human review request for historical excerpts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    raw_data_path,
    workspace_path,
)


DEFAULT_RAW = raw_data_path("architecture_migration", "source_rows.jsonl")
DEFAULT_PROVENANCE = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_provenance_review.json"
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_human_review_request.json"
)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return value


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def build_request(
    raw_rows: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> Dict[str, Any]:
    targets = provenance.get("manual_review_targets", [])
    if not isinstance(targets, list) or len(targets) != 6:
        raise ValueError("provenance report must expose exactly six manual review targets")
    raw_by_id = {str(row.get("record_id", "")): row for row in raw_rows}
    reviews: List[Dict[str, Any]] = []
    for target in targets:
        record_id = str(target.get("record_id", ""))
        row = raw_by_id.get(record_id)
        if row is None:
            raise ValueError(f"manual review target is missing from raw evidence: {record_id}")
        content = str(row.get("content", ""))
        source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if (
            str(row.get("content_origin", "")) != "transcribed_source_excerpt"
            or source_hash != str(row.get("source_hash", ""))
            or source_hash != str(target.get("source_hash", ""))
        ):
            raise ValueError(f"manual review target failed content binding: {record_id}")
        reviews.append(
            {
                "record_id": record_id,
                "source_url": str(row.get("source_url", "")),
                "source_ref": str(row.get("source_ref") or row.get("source_url", "")),
                "source_revision": str(row.get("source_revision", "")),
                "source_hash": source_hash,
                "stored_excerpt": content,
                "review_status": "pending_human_review",
                "required_review": {
                    "authoritative_section_locator": "",
                    "authoritative_text_hash": "",
                    "alignment_decision": "pending",
                    "semantic_omission_or_distortion_found": None,
                    "reviewer": "",
                    "reviewed_at": "",
                    "notes": "",
                },
            }
        )
    return {
        "schema": "sara-phase34-transcribed-excerpt-human-review-request-v1",
        "observed_only": True,
        "review_complete": False,
        "promotion_ready": False,
        "target_count": len(reviews),
        "targets": reviews,
        "completion_rule": (
            "Every target requires an explicit human alignment decision, authoritative "
            "section locator and hash, reviewer identity, review time, and no unresolved distortion."
        ),
        "mutation_policy": {
            "historical_raw_rows_mutated": False,
            "executed_v2_fingerprint_mutated": False,
            "silent_reclassification_allowed": False,
            "replacement_requires_new_preregistration": True,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", default=DEFAULT_RAW)
    parser.add_argument("--provenance-path", default=DEFAULT_PROVENANCE)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        request = build_request(_read_jsonl(args.raw_path), _read_json(args.provenance_path))
        with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
            json.dump(request, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema": request["schema"],
                "review_complete": request["review_complete"],
                "target_count": request["target_count"],
                "output_path": os.path.realpath(args.output_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
