#!/usr/bin/env python3
"""Freeze approved Phase 37 sources and base structural split fixtures."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import os
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402

DEFAULT_DRAFT = workspace_path("evaluation", "phase37_source_split_review_draft.json")
DEFAULT_DECISIONS = workspace_path("evaluation", "phase37_source_split_human_review_decisions.json")
DEFAULT_SOURCE_MANIFEST = processed_data_path("autobot", "phase37_structural_source_manifest.jsonl")
DEFAULT_TRAIN_FIXTURE = processed_data_path("benchmark_fixtures", "phase37_structural_train_base.jsonl")
DEFAULT_EVALUATION_FIXTURE = processed_data_path("benchmark_fixtures", "phase37_structural_evaluation_base.jsonl")
DEFAULT_RECEIPT = workspace_path("evaluation", "phase37_source_split_freeze_receipt.json")


def _read(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("freeze input must be an object")
    return value


def _review_module():
    path = os.path.join(PROJECT_ROOT, "scripts", "eval", "phase37_source_split_review.py")
    spec = importlib.util.spec_from_file_location("phase37_source_review_freeze", path)
    if spec is None or spec.loader is None:
        raise ValueError("unable to load Phase 37 review gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)


def _sha(payload: str) -> str:
    return sha256(payload.encode("utf-8")).hexdigest()


def build_artifacts(draft: Mapping[str, Any], decisions: Mapping[str, Any]) -> Dict[str, Any]:
    packet = _review_module().build_packet(draft, decisions)
    if decisions.get("draft_fingerprint") != packet["draft_fingerprint"]:
        raise ValueError("human decisions do not bind the current draft fingerprint")
    if not packet["fixture_freeze_allowed"]:
        raise ValueError("Phase 37 fixture freeze gate is closed")
    sources = list(draft["sources"])
    source_rows = []
    fixture_rows = {"train": [], "evaluation": []}
    for source in sources:
        evidence_id = f"evidence::{source['source_hash'][:16]}"
        source_rows.append({
            "schema": "sara-phase37-structural-source-v1",
            "record_id": source["record_id"],
            "partition": source["partition"],
            "structural_family": source["structural_family"],
            "semantic_domain": source["semantic_domain"],
            "source_url": source["source_url"],
            "source_hash": source["source_hash"],
            "excerpt": source["excerpt"],
            "evidence_id": evidence_id,
            "human_reviewed": True,
            "observed_only": True,
        })
        edges = [
            {"source": edge[0], "relation_type": edge[1], "target": edge[2], "evidence_id": evidence_id, "verified": True}
            for edge in source["proposed_edges"]
        ]
        fixture_rows[source["partition"]].append({
            "schema": "sara-phase37-structural-base-case-v1",
            "case_id": source["record_id"],
            "partition": source["partition"],
            "structural_family": source["structural_family"],
            "semantic_domain": source["semantic_domain"],
            "visible_edges": edges if source["partition"] == "train" else edges[:-1],
            "withheld_edge": None if source["partition"] == "train" else edges[-1],
            "evidence_ids": [evidence_id],
            "direct_durable_mutation_allowed": False,
        })
    payloads = {
        "source_manifest": _jsonl(source_rows),
        "train_fixture": _jsonl(fixture_rows["train"]),
        "evaluation_fixture": _jsonl(fixture_rows["evaluation"]),
    }
    return {"packet": packet, "payloads": payloads, "hashes": {key: _sha(value) for key, value in payloads.items()}}


def _write_new_or_identical(path: str, payload: str) -> str:
    resolved = ensure_parent_directory(path)
    if os.path.exists(resolved):
        with open(resolved, encoding="utf-8") as handle:
            if handle.read() != payload:
                raise ValueError(f"frozen artifact is immutable: {path}")
        return "identical_preserved"
    with open(resolved, "x", encoding="utf-8") as handle:
        handle.write(payload)
    return "created"


def freeze(draft_path: str, decisions_path: str, source_path: str, train_path: str, evaluation_path: str, receipt_path: str) -> Dict[str, Any]:
    artifacts = build_artifacts(_read(draft_path), _read(decisions_path))
    paths = {"source_manifest": source_path, "train_fixture": train_path, "evaluation_fixture": evaluation_path}
    {key: _write_new_or_identical(paths[key], artifacts["payloads"][key]) for key in paths}
    receipt = {
        "schema": "sara-phase37-source-split-freeze-receipt-v1",
        "protocol_fingerprint": artifacts["packet"]["protocol_fingerprint"],
        "review_draft_fingerprint": artifacts["packet"]["draft_fingerprint"],
        "human_review_complete": True,
        "fixture_freeze_allowed": True,
        "source_count": artifacts["packet"]["source_count"],
        "train_count": artifacts["packet"]["train_count"],
        "evaluation_count": artifacts["packet"]["evaluation_count"],
        "artifact_hashes": artifacts["hashes"],
        "artifact_paths": {key: os.path.realpath(value) for key, value in paths.items()},
        "statuses": {key: "frozen" for key in paths},
        "candidate_implementation_allowed": True,
        "production_promotion_allowed": False,
    }
    receipt_payload = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_new_or_identical(receipt_path, receipt_payload)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-path", default=DEFAULT_DRAFT)
    parser.add_argument("--decisions-path", default=DEFAULT_DECISIONS)
    parser.add_argument("--source-path", default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_FIXTURE)
    parser.add_argument("--evaluation-path", default=DEFAULT_EVALUATION_FIXTURE)
    parser.add_argument("--receipt-path", default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    try:
        receipt = freeze(args.draft_path, args.decisions_path, args.source_path, args.train_path, args.evaluation_path, args.receipt_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
