#!/usr/bin/env python3
"""Replay independent external histories through real subsystem adapters."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from hashlib import sha256
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.edge.portable_decision_trace import (  # noqa: E402
    adapt_event_memory_admission,
    adapt_event_memory_evictions,
    adapt_event_memory_retrieval,
    adapt_event_memory_revision,
    adapt_predictive_feedback,
    adapt_risa_proposal,
    canonical_decision_json,
    decision_trace_digest,
)
from sara_engine.memory.event_state_cache import (  # noqa: E402
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.memory.verification_receipt import issue_verification_receipt  # noqa: E402
from sara_engine.risa.structural_interpolation import (  # noqa: E402
    PredictiveStructuralFeedbackEngine,
    StructuralEvidence,
    StructuralFeedbackSignal,
    StructuralInterpolationEngine,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_SOURCE = processed_data_path("autobot", "architecture_migration_external_manifest.jsonl")
DEFAULT_OUTPUT = workspace_path("evaluation", "phase27_independent_decision_replay.json")


def load_rows(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def build_report(rows: Sequence[Mapping[str, Any]], rust_core: Any | None = None) -> Dict[str, Any]:
    source_checks = {
        "six_rows_present": len(rows) == 6,
        "independent_external_scope": all(row.get("evidence_scope") == "independent_external" for row in rows),
        "provenance_present": all(row.get("provenance_digest") and row.get("source_revision") for row in rows),
        "two_source_domains_present": len({str(row.get("source_domain", "")) for row in rows}) == 2,
        "ordered_horizon": [int(row.get("migration_horizon_index", -1)) for row in rows] == list(range(len(rows))),
    }
    cache = VerifiedHierarchicalEventStateCache(max_entries=4, retention_profile="fixed")
    records: List[Dict[str, Any]] = []
    next_sequence = 0
    for source_index, row in enumerate(rows):
        source_ref = str(row["source_ref"])
        source_revision = str(row["source_revision"])
        entry_id = str(row["manifest_id"])
        receipt = issue_verification_receipt(
            verifier_id="phase27-independent-source-gate",
            verifier_version="v1",
            decision="verified_external_manifest",
            evidence={"material_hash": row["material_hash"], "provenance_digest": row["provenance_digest"]},
            source_refs=(source_ref,),
            source_revision=source_revision,
            observed=True,
            source_backed=True,
            verified=True,
        )
        result = cache.admit(
            EventStateCandidate(
                entry_id=entry_id,
                signature=tuple(int(item) for item in row["sparse_signature"]),
                source_ref=source_ref,
                source_revision=source_revision,
                time_segment=source_index,
                own_latent_id=str(row["latent_cluster_id"]),
                confidence=float(row["quality_score"]),
                uncertainty=0.0,
                source_reliability=1.0,
                resonance_score=0.9,
                metabolic_headroom=0.9,
                observed=True,
                source_backed=True,
                verified=True,
                event_cost=int(row["event_cost"]),
                verification_receipt=receipt,
            )
        )
        records.append(adapt_event_memory_admission(result, sequence=next_sequence))
        next_sequence += 1
        eviction_records = adapt_event_memory_evictions(
            result, sequence_start=next_sequence
        )
        records.extend(eviction_records)
        next_sequence += len(eviction_records)

    for row in rows:
        retrieval = cache.retrieve(
            tuple(int(item) for item in row["sparse_signature"]),
            own_latent_id=str(row["latent_cluster_id"]),
            source_ref=str(row["source_ref"]),
            top_k=1,
        )
        records.append(
            adapt_event_memory_retrieval(
                retrieval,
                subject_id=str(row["manifest_id"]),
                sequence=next_sequence,
            )
        )
        next_sequence += 1

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_domain"]), []).append(row)
    for domain, items in sorted(grouped.items()):
        evidence = tuple(
            StructuralEvidence(
                source_node=f"source-domain:{domain}",
                target_node="documentation-section",
                relation_type="contains",
                confidence=float(item["quality_score"]),
                source_ref=str(item["source_ref"]),
                source_hash=str(item["material_hash"]),
                source_revision=str(item["source_revision"]),
                acquired_at=int(item["migration_horizon_index"]),
                verified=True,
            )
            for item in items
        )
        result = StructuralInterpolationEngine().propose(evidence)
        for proposal in result.proposals:
            records.append(adapt_risa_proposal(proposal, sequence=next_sequence))
            next_sequence += 1

    feedback = PredictiveStructuralFeedbackEngine(mismatch_threshold=0.25)
    for previous, current in zip(rows, rows[1:]):
        predicted = _jaccard(previous["sparse_signature"], current["sparse_signature"])
        observed = float(previous["source_domain"] == current["source_domain"])
        proposal = feedback.propose(
            (
                StructuralFeedbackSignal(
                    predicting_concept="source-continuity",
                    source_node=str(previous["manifest_id"]),
                    target_node=str(current["manifest_id"]),
                    relation_type="followed_by",
                    predicted_confidence=predicted,
                    observed_confidence=observed,
                    evidence_ids=(str(current["provenance_digest"]),),
                ),
            )
        )[0]
        records.append(adapt_predictive_feedback(proposal, sequence=next_sequence))
        next_sequence += 1

    python_json = canonical_decision_json(records)
    python_digest = decision_trace_digest(records)
    canonical_fn = getattr(rust_core, "canonical_portable_decision_trace_json", None)
    digest_fn = getattr(rust_core, "portable_decision_trace_digest", None)
    rust_available = callable(canonical_fn) and callable(digest_fn)
    source = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    rust_json = str(canonical_fn(source)) if rust_available else ""
    rust_digest = str(digest_fn(source)) if rust_available else ""
    controlled_records: List[Dict[str, Any]] = []
    if rows:
        base = rows[0]
        control_cache = VerifiedHierarchicalEventStateCache(max_entries=2, retention_profile="fixed")

        def controlled_candidate(revision: str, segment: int, contradicted: bool = False) -> EventStateCandidate:
            receipt = issue_verification_receipt(
                verifier_id="phase27-controlled-revision-gate",
                verifier_version="v1",
                decision="verified_controlled_perturbation",
                evidence={"base_material_hash": base["material_hash"], "revision": revision},
                source_refs=(str(base["source_ref"]),),
                source_revision=revision,
                observed=True,
                source_backed=True,
                verified=True,
                contradicted=contradicted,
            )
            return EventStateCandidate(
                entry_id=f"controlled::{revision}",
                signature=tuple(int(item) for item in base["sparse_signature"]),
                source_ref=str(base["source_ref"]),
                source_revision=revision,
                time_segment=segment,
                own_latent_id="controlled-revision",
                confidence=1.0,
                uncertainty=0.0,
                source_reliability=1.0,
                resonance_score=0.9,
                metabolic_headroom=0.9,
                observed=True,
                source_backed=True,
                verified=True,
                contradicted=contradicted,
                verification_receipt=receipt,
            )

        control_cache.admit(controlled_candidate("controlled-r1", 100))
        replacement = control_cache.admit(controlled_candidate("controlled-r2", 101))
        controlled_records.append(adapt_event_memory_revision(replacement, sequence=0))
        contradiction = control_cache.admit(
            controlled_candidate("controlled-r3-contradiction", 102, contradicted=True)
        )
        controlled_records.append(adapt_event_memory_revision(contradiction, sequence=1))
        oscillation = feedback.propose(
            (
                StructuralFeedbackSignal(
                    predicting_concept="controlled-source-continuity",
                    source_node=str(base["manifest_id"]),
                    target_node=str(base["manifest_id"]),
                    relation_type="revision_cycle",
                    predicted_confidence=0.2,
                    observed_confidence=0.9,
                    evidence_ids=(str(base["provenance_digest"]),),
                    recent_actions=(
                        "strengthen_relation",
                        "cut_relation",
                        "strengthen_relation",
                        "cut_relation",
                    ),
                ),
            )
        )[0]
        controlled_records.append(adapt_predictive_feedback(oscillation, sequence=2))
    controlled_source = json.dumps(controlled_records, ensure_ascii=True, separators=(",", ":"))
    controlled_python_json = canonical_decision_json(controlled_records)
    controlled_python_digest = decision_trace_digest(controlled_records)
    controlled_rust_json = str(canonical_fn(controlled_source)) if rust_available else ""
    controlled_rust_digest = str(digest_fn(controlled_source)) if rust_available else ""
    checks = {
        **source_checks,
        "bounded_cache_state": len(cache.entries) <= 4,
        "event_memory_steps_present": sum(row["subsystem"] == "event_memory" for row in records) == 6,
        "retrieval_steps_present": sum(row["subsystem"] == "event_memory_retrieval" for row in records) == 6,
        "eviction_steps_present": sum(row["subsystem"] == "event_memory_eviction" for row in records) == 2,
        "risa_steps_present": sum(row["subsystem"] == "risa_proposal" for row in records) == 2,
        "predictive_steps_present": sum(row["subsystem"] == "predictive_feedback" for row in records) == 5,
        "rust_extension_available": rust_available,
        "canonical_bytes_equivalent": rust_available and rust_json == python_json,
        "digest_equivalent": rust_available and rust_digest == python_digest,
        "controlled_revision_replaced": bool(
            controlled_records and controlled_records[0]["prediction_match"] is False
        ),
        "controlled_contradiction_frozen": bool(
            len(controlled_records) > 1 and controlled_records[1]["contradiction"] is True
        ),
        "controlled_feedback_oscillation_frozen": bool(
            len(controlled_records) > 2 and controlled_records[2]["contradiction"] is True
        ),
        "controlled_bytes_equivalent": rust_available and controlled_rust_json == controlled_python_json,
        "controlled_digest_equivalent": rust_available and controlled_rust_digest == controlled_python_digest,
    }
    return {
        "schema": "sara-phase27-independent-decision-replay-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "production_path_changed": False,
        "independent_evidence": True,
        "source_manifest_fingerprint": _fingerprint(list(rows)),
        "decision_trace_digest": python_digest,
        "rust_decision_trace_digest": rust_digest or None,
        "controlled_perturbation": {
            "scope": "synthetic_control_over_external_base_record",
            "decision_count": len(controlled_records),
            "decision_trace_digest": controlled_python_digest,
            "rust_decision_trace_digest": controlled_rust_digest or None,
            "independent_evidence": False,
        },
        "checks": checks,
        "metrics": {
            "source_row_count": len(rows),
            "decision_count": len(records),
            "cache_entry_count": len(cache.entries),
            "cache_eviction_count": cache.eviction_count,
        },
        "claim_boundary": "Independent source records for the base run; revision, contradiction, and oscillation are explicitly synthetic controls over one external base record; no semantic accuracy or full-subsystem equivalence claim.",
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
