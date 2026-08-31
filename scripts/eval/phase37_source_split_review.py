#!/usr/bin/env python3
"""Build and gate the Phase 37 human source/split review packet."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, raw_data_path, workspace_path  # noqa: E402

DEFAULT_DRAFT = workspace_path("evaluation", "phase37_source_split_review_draft.json")
DEFAULT_DECISIONS = workspace_path("evaluation", "phase37_source_split_human_review_decisions.json")
DEFAULT_PACKET = workspace_path("evaluation", "phase37_source_split_review_packet.json")
RAW_SOURCES = raw_data_path("architecture_migration", "source_rows.jsonl")
PROTOCOL = "e77d34460bfc2ae2440d765616a65ce7dad734d07ef6cca3b0d17b1532cfe704"


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("review input must be a JSON object")
    return value


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_packet(draft: Mapping[str, Any], decisions: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    raw = {}
    with open(RAW_SOURCES, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            raw[row["source_url"]] = row
    sources = list(draft.get("sources", ()))
    source_checks = []
    for source in sources:
        stored = raw.get(source["source_url"])
        source_checks.append({
            "record_id": source["record_id"],
            "raw_source_present": stored is not None,
            "source_hash_matches": bool(stored and stored.get("source_hash") == source.get("source_hash")),
            "excerpt_present_in_snapshot": bool(stored and source.get("excerpt") in stored.get("content", "")),
            "edge_count": len(source.get("proposed_edges", ())),
        })
    train = [row for row in sources if row.get("partition") == "train"]
    evaluation = [row for row in sources if row.get("partition") == "evaluation"]
    train_families = {row["structural_family"] for row in train}
    evaluation_families = {row["structural_family"] for row in evaluation}
    train_refs = {row["source_url"] for row in train}
    evaluation_refs = {row["source_url"] for row in evaluation}
    automated = {
        "protocol_matches": draft.get("protocol_fingerprint") == PROTOCOL,
        "minimum_eight_sources": len(sources) >= 8,
        "unique_source_urls": len({row["source_url"] for row in sources}) == len(sources),
        "minimum_four_structural_families": len({row["structural_family"] for row in sources}) >= 4,
        "train_evaluation_sources_disjoint": not (train_refs & evaluation_refs),
        "train_evaluation_families_disjoint": not (train_families & evaluation_families),
        "heldout_semantic_domain_present": bool({row["semantic_domain"] for row in evaluation} - {row["semantic_domain"] for row in train}),
        "all_raw_bindings_valid": all(item["raw_source_present"] and item["source_hash_matches"] and item["excerpt_present_in_snapshot"] and item["edge_count"] > 0 for item in source_checks),
    }
    decision_rows = (decisions or {}).get("decisions", [])
    approved = {row.get("record_id") for row in decision_rows if row.get("decision") == "approve" and row.get("reviewer") == "human_operator"}
    human_review_complete = len(approved) == len(sources) and approved == {row["record_id"] for row in sources}
    return {
        "schema": "sara-phase37-source-split-review-packet-v1",
        "protocol_fingerprint": PROTOCOL,
        "draft_fingerprint": _digest(draft),
        "automated_checks": automated,
        "automated_integrity_passed": all(automated.values()),
        "human_review_complete": human_review_complete,
        "fixture_freeze_allowed": all(automated.values()) and human_review_complete,
        "source_count": len(sources),
        "train_count": len(train),
        "evaluation_count": len(evaluation),
        "source_checks": source_checks,
        "review_questions": draft.get("review_questions", []),
        "sources": sources,
        "claim_boundary": "Automated checks establish snapshot identity and split integrity only. Semantic edge alignment requires explicit human approval for every row.",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-path", default=DEFAULT_DRAFT)
    parser.add_argument("--decisions-path", default=DEFAULT_DECISIONS)
    parser.add_argument("--packet-path", default=DEFAULT_PACKET)
    args = parser.parse_args(argv)
    draft = _read_json(args.draft_path)
    decisions = _read_json(args.decisions_path) if os.path.exists(args.decisions_path) else None
    packet = build_packet(draft, decisions)
    with open(ensure_parent_directory(args.packet_path), "w", encoding="utf-8") as handle:
        json.dump(packet, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: packet[key] for key in ("automated_integrity_passed", "human_review_complete", "fixture_freeze_allowed", "draft_fingerprint")}, indent=2))
    return 0 if packet["automated_integrity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
