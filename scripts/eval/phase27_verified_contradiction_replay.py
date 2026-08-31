#!/usr/bin/env python3
"""Replay an original RFC claim against its verified technical correction."""

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
    decide,
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

DEFAULT_SOURCE = processed_data_path("autobot", "phase27_verified_contradiction_manifest.jsonl")
DEFAULT_OUTPUT = workspace_path("evaluation", "phase27_verified_contradiction_replay.json")


def load_rows(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _candidate(row: Mapping[str, Any]) -> EventStateCandidate:
    contradicted = row["polarity"] == "contradicts"
    receipt = issue_verification_receipt(
        verifier_id="phase27-rfc-verified-erratum-gate",
        verifier_version="v1",
        decision="verified_technical_erratum" if contradicted else "published_original",
        evidence={
            "claim_value": row["claim_value"],
            "material_hash": row["material_hash"],
            "source_role": row["source_role"],
        },
        source_refs=(str(row["logical_source_ref"]), str(row["source_url"])),
        source_revision=str(row["source_revision"]),
        observed=True,
        source_backed=True,
        verified=True,
        contradicted=contradicted,
    )
    return EventStateCandidate(
        entry_id=str(row["claim_id"]),
        signature=tuple(int(value) for value in row["sparse_signature"]),
        source_ref=str(row["logical_source_ref"]),
        source_revision=str(row["source_revision"]),
        time_segment=int(row["time_segment"]),
        own_latent_id=str(row["proposition_id"]),
        confidence=1.0,
        uncertainty=0.0,
        source_reliability=1.0,
        resonance_score=1.0,
        metabolic_headroom=1.0,
        observed=True,
        source_backed=True,
        verified=True,
        contradicted=contradicted,
        event_cost=1,
        verification_receipt=receipt,
    )


def build_report(rows: Sequence[Mapping[str, Any]], rust_core: Any | None = None) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["time_segment"]))
    cache = VerifiedHierarchicalEventStateCache(max_entries=2, retention_profile="fixed")
    results = [cache.admit(_candidate(row)) for row in ordered]
    record = adapt_event_memory_revision(results[-1], sequence=0) if results else {}
    records = [record] if record else []
    python_json = canonical_decision_json(records)
    python_digest = decision_trace_digest(records)
    canonical_fn = getattr(rust_core, "canonical_portable_decision_trace_json", None)
    digest_fn = getattr(rust_core, "portable_decision_trace_digest", None)
    rust_available = callable(canonical_fn) and callable(digest_fn)
    source = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    rust_json = str(canonical_fn(source)) if rust_available else ""
    rust_digest = str(digest_fn(source)) if rust_available else ""
    state = cache.state_dict()
    checks = {
        "two_ordered_claims": len(ordered) == 2
        and [int(row["time_segment"]) for row in ordered] == [0, 1],
        "same_proposition": len({str(row["proposition_id"]) for row in ordered}) == 1,
        "separate_source_records": len({str(row["source_url"]) for row in ordered}) == 2
        and len({str(row["material_hash"]) for row in ordered}) == 2,
        "opposite_polarities": [row["polarity"] for row in ordered]
        == ["supports", "contradicts"]
        and [bool(row["claim_value"]) for row in ordered] == [True, False],
        "verified_correction_role": len(ordered) == 2
        and ordered[1]["source_role"] == "verified_technical_erratum",
        "independent_external_scope": all(
            row.get("evidence_scope") == "independent_external_verified_contradiction"
            for row in ordered
        ),
        "original_admitted": len(results) == 2 and results[0].accepted is True,
        "contradiction_frozen": len(results) == 2
        and results[1].accepted is False
        and results[1].decision == "block_contradiction",
        "original_state_preserved": len(state["entries"]) == 1
        and state["entries"][0]["source_revision"] == ordered[0]["source_revision"],
        "portable_freeze_decision": bool(record)
        and decide(record) == "freeze_revision",
        "rust_extension_available": rust_available,
        "canonical_bytes_equivalent": rust_available and rust_json == python_json,
        "digest_equivalent": rust_available and rust_digest == python_digest,
    }
    return {
        "schema": "sara-phase27-verified-contradiction-replay-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "production_path_changed": False,
        "independent_evidence": True,
        "decision_trace_digest": python_digest,
        "rust_decision_trace_digest": rust_digest or None,
        "checks": checks,
        "metrics": {"claim_count": len(ordered), "decision_count": len(records)},
        "claim_boundary": "The original RFC ABNF and its Verified Technical Erratum prove one explicit permission-versus-exclusion conflict. This does not establish general contradiction detection or semantic reasoning.",
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
