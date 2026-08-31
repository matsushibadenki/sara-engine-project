#!/usr/bin/env python3
"""Replay a genuine external source revision through the portable decision boundary."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.edge.portable_decision_trace import (  # noqa: E402
    adapt_event_memory_revision,
    canonical_decision_json,
    decision_trace_digest,
)
from sara_engine.memory.event_state_cache import (  # noqa: E402
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.memory.verification_receipt import issue_verification_receipt  # noqa: E402
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_SOURCE = processed_data_path("autobot", "phase27_revision_history_manifest.jsonl")
DEFAULT_OUTPUT = workspace_path("evaluation", "phase27_revision_history_replay.json")


def load_rows(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _candidate(row: Mapping[str, Any]) -> EventStateCandidate:
    receipt = issue_verification_receipt(
        verifier_id="phase27-independent-revision-gate",
        verifier_version="v1",
        decision="verified_external_source_revision",
        evidence={
            "material_hash": row["material_hash"],
            "source_url": row["source_url"],
        },
        source_refs=(str(row["logical_source_ref"]), str(row["source_url"])),
        source_revision=str(row["source_revision"]),
        observed=True,
        source_backed=True,
        verified=True,
    )
    return EventStateCandidate(
        entry_id=str(row["manifest_id"]),
        signature=tuple(int(value) for value in row["sparse_signature"]),
        source_ref=str(row["logical_source_ref"]),
        source_revision=str(row["source_revision"]),
        time_segment=int(row["time_segment"]),
        own_latent_id="cpython-argparse-logical-source",
        confidence=1.0,
        uncertainty=0.0,
        source_reliability=1.0,
        resonance_score=1.0,
        metabolic_headroom=1.0,
        observed=True,
        source_backed=True,
        verified=True,
        event_cost=1,
        verification_receipt=receipt,
    )


def build_report(rows: Sequence[Mapping[str, Any]], rust_core: Any | None = None) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["time_segment"]))
    cache = VerifiedHierarchicalEventStateCache(max_entries=2, retention_profile="fixed")
    results = [cache.admit(_candidate(row)) for row in ordered]
    revision_records = [
        adapt_event_memory_revision(result, sequence=index)
        for index, result in enumerate(results[1:])
    ]
    python_json = canonical_decision_json(revision_records)
    python_digest = decision_trace_digest(revision_records)
    canonical_fn = getattr(rust_core, "canonical_portable_decision_trace_json", None)
    digest_fn = getattr(rust_core, "portable_decision_trace_digest", None)
    rust_available = callable(canonical_fn) and callable(digest_fn)
    source = json.dumps(revision_records, ensure_ascii=True, separators=(",", ":"))
    rust_json = str(canonical_fn(source)) if rust_available else ""
    rust_digest = str(digest_fn(source)) if rust_available else ""
    state = cache.state_dict()
    checks = {
        "two_ordered_revisions": len(ordered) == 2
        and [int(row["time_segment"]) for row in ordered] == [0, 1],
        "same_logical_source": len({str(row["logical_source_ref"]) for row in ordered}) == 1,
        "distinct_materials": len({str(row["material_hash"]) for row in ordered}) == 2,
        "independent_external_scope": all(
            row.get("evidence_scope") == "independent_external_version_history"
            for row in ordered
        ),
        "verified_revision_replaced": len(results) == 2
        and results[1].decision == "replace_verified_revision",
        "latest_revision_retained": bool(state["entries"])
        and state["entries"][0]["source_revision"] == ordered[-1]["source_revision"],
        "portable_replace_decision": len(revision_records) == 1
        and revision_records[0]["prediction_match"] is False,
        "rust_extension_available": rust_available,
        "canonical_bytes_equivalent": rust_available and rust_json == python_json,
        "digest_equivalent": rust_available and rust_digest == python_digest,
    }
    return {
        "schema": "sara-phase27-revision-history-replay-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "production_path_changed": False,
        "independent_evidence": True,
        "decision_trace_digest": python_digest,
        "rust_decision_trace_digest": rust_digest or None,
        "checks": checks,
        "metrics": {"source_revision_count": len(ordered), "decision_count": len(revision_records)},
        "claim_boundary": "Official CPython tag and content-hash evidence proves one source revision replacement only; it does not prove semantic contradiction, general migration accuracy, or production readiness.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", default=DEFAULT_SOURCE)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        rust_core = importlib.import_module("sara_engine.sara_rust_core")
    except ImportError:
        rust_core = None
    report = build_report(load_rows(args.source_path), rust_core)
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
