#!/usr/bin/env python3
"""Run a frozen source-isolated Event Memory architecture-migration benchmark."""

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

from sara_engine.ingest import make_candidate_relation  # noqa: E402
from sara_engine.memory.architecture_migration import (  # noqa: E402
    ArchitectureMigrationCoordinator,
    ArchitectureMigrationPolicy,
)
from sara_engine.memory.concept_admission import ConceptRevalidationEntry  # noqa: E402
from sara_engine.memory.concept_review_loop import ConceptReviewLoop  # noqa: E402
from sara_engine.memory.event_state_cache import (  # noqa: E402
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.risa import (  # noqa: E402
    SARAAlignedRisaKernel,
    ingest_verified_surface_into_risa,
)
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_REPORT_PATH = workspace_path("evaluation", "architecture_migration_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path(
    "evaluation", "architecture_migration_benchmark_summary.txt"
)


def _frozen_workload() -> List[Dict[str, Any]]:
    return [
        {
            "entry_id": "source-a-cue-response",
            "signature": (11, 13, 17),
            "source_ref": "benchmark://source-a",
            "source_revision": "source-a-revision-1",
            "own_latent_id": "predicts:vision:cue->audio:response",
            "causal_predecessors": ("vision:cue", "audio:response"),
            "architecture_version": "sara-architecture-v1",
        },
        {
            "entry_id": "source-b-cue-response",
            "signature": (19, 23, 29),
            "source_ref": "benchmark://source-b",
            "source_revision": "source-b-revision-4",
            "own_latent_id": "predicts:vision:cue->audio:response",
            "causal_predecessors": ("vision:cue", "audio:response"),
            "architecture_version": "sara-architecture-v1",
        },
        {
            "entry_id": "source-c-context-response",
            "signature": (31, 37, 41),
            "source_ref": "benchmark://source-c",
            "source_revision": "source-c-revision-2",
            "own_latent_id": "predicts:text:context->audio:response",
            "causal_predecessors": ("text:context", "audio:response"),
            "architecture_version": "sara-architecture-v1",
        },
        {
            "entry_id": "incompatible-legacy-route",
            "signature": (43, 47, 53),
            "source_ref": "benchmark://legacy-incompatible",
            "source_revision": "legacy-layout-0",
            "own_latent_id": "observes:legacy:route->audio:response",
            "causal_predecessors": ("legacy:route",),
            "architecture_version": "sara-architecture-v0",
        },
    ]


def _candidate(row: Mapping[str, Any], *, time_segment: int) -> EventStateCandidate:
    return EventStateCandidate(
        entry_id=str(row["entry_id"]),
        signature=tuple(int(value) for value in row["signature"]),
        source_ref=str(row["source_ref"]),
        source_revision=str(row["source_revision"]),
        time_segment=time_segment,
        own_latent_id=str(row["own_latent_id"]),
        causal_predecessors=tuple(str(value) for value in row["causal_predecessors"]),
        confidence=0.91,
        uncertainty=0.09,
        source_reliability=0.9,
        resonance_score=0.9,
        sequence_support_score=0.7,
        sequence_support_count=3,
        credit_score=0.76,
        credit_responsibility=0.78,
        credit_confidence=0.82,
        credit_longevity=0.86,
        metabolic_headroom=0.9,
        observed=True,
        source_backed=True,
        verified=True,
        event_cost=4,
        architecture_version=str(row["architecture_version"]),
    )


def _concept_review() -> Any:
    concept_key = "predicts:vision:cue->audio:response"
    queue = (
        ConceptRevalidationEntry(
            concept_key=concept_key,
            decision="quarantine_counterexample_pressure",
            supporting_relation_ids=(concept_key,),
            source_refs=("benchmark://source-a",),
            source_hashes=("source-a-revision-1",),
            revision_conflict_count=0,
            contradiction_score=0.45,
            next_action="collect_counterexamples_and_retest",
            blocked_at_segment=1,
            last_review_segment=1,
            retry_after_segment=2,
        ),
    )
    relations = tuple(
        make_candidate_relation(
            {
                "record_id": f"migration-relation-{index}",
                "relation": "predicts",
                "source_event_id": "vision:cue",
                "target_event_id": "audio:response",
                "delay_lower_ms": 20,
                "delay_upper_ms": 60,
                "confidence": 0.9,
                "source_ref": source_ref,
                "source_hash": source_hash,
                "extractor_name": "architecture_migration_fixture",
                "extractor_version": "v1",
                "evidence_count": 3,
                "counterexample_count": 0,
                "prediction_gain": 0.3,
            }
        )
        for index, (source_ref, source_hash) in enumerate(
            (
                ("benchmark://source-a", "source-a-revision-1"),
                ("benchmark://source-b", "source-b-revision-4"),
            )
        )
    )
    return ConceptReviewLoop().run(queue, relations, current_segment=4)


def build_report() -> Dict[str, Any]:
    workload = _frozen_workload()
    workload_bytes = json.dumps(workload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    legacy_cache = VerifiedHierarchicalEventStateCache(
        retention_profile="fixed",
        max_entries=6,
        retrieval_threshold=0.2,
    )
    for segment, row in enumerate(workload, start=1):
        legacy_cache.admit(_candidate(row, time_segment=segment))
    legacy_before = legacy_cache.state_dict()
    target_cache = VerifiedHierarchicalEventStateCache(
        retention_profile="fixed",
        max_entries=6,
        retrieval_threshold=0.2,
    )
    migration = ArchitectureMigrationCoordinator(
        ArchitectureMigrationPolicy(
            source_architecture_version="sara-architecture-v1",
            target_architecture_version="sara-architecture-v2",
            max_replay_candidates=3,
        )
    ).migrate(legacy_cache, target_cache)
    legacy_unchanged_after_migration = legacy_cache.state_dict() == legacy_before
    migrated_rows = [row for row in workload if row["architecture_version"] == "sara-architecture-v1"]
    legacy_recall = 0
    target_recall = 0
    retrieval_trace: List[Dict[str, Any]] = []
    for row in migrated_rows:
        legacy = legacy_cache.retrieve(
            row["signature"],
            own_latent_id=str(row["own_latent_id"]),
            source_ref=str(row["source_ref"]),
            now_segment=32,
        )
        target = target_cache.retrieve(
            row["signature"],
            own_latent_id=str(row["own_latent_id"]),
            source_ref=str(row["source_ref"]),
            now_segment=32,
        )
        expected_target_id = f"migration:sara-architecture-v2:{row['entry_id']}"
        legacy_matched = bool(legacy.matches) and legacy.matches[0]["entry_id"] == row["entry_id"]
        target_matched = bool(target.matches) and target.matches[0]["entry_id"] == expected_target_id
        legacy_recall += int(legacy_matched)
        target_recall += int(target_matched)
        retrieval_trace.append(
            {
                "source_ref": row["source_ref"],
                "legacy_matched": legacy_matched,
                "target_matched": target_matched,
                "legacy_event_cost": legacy.event_cost,
                "target_event_cost": target.event_cost,
            }
        )
    target_candidates = [
        EventStateCandidate(
            entry_id=entry.entry_id,
            signature=entry.signature,
            source_ref=entry.source_ref,
            source_revision=entry.source_revision,
            time_segment=entry.time_segment,
            own_latent_id=entry.own_latent_id,
            causal_predecessors=entry.causal_predecessors,
            confidence=entry.confidence,
            uncertainty=entry.uncertainty,
            source_reliability=entry.source_reliability,
            resonance_score=entry.resonance_score,
            sequence_support_score=entry.sequence_support_score,
            sequence_support_count=entry.sequence_support_count,
            credit_score=entry.credit_score,
            credit_responsibility=entry.credit_responsibility,
            credit_confidence=entry.credit_confidence,
            credit_longevity=entry.credit_longevity,
            metabolic_headroom=1.0,
            observed=entry.observed,
            source_backed=True,
            verified=entry.verified,
            event_cost=entry.event_cost,
            architecture_version=entry.architecture_version,
            migration_source_architecture_version=entry.migration_source_architecture_version,
        )
        for entry in target_cache.entries.values()
    ]
    kernel = SARAAlignedRisaKernel(min_support=2, min_distinct_actors=2)
    risa_observations = ingest_verified_surface_into_risa(
        kernel,
        event_state_candidates=target_candidates,
    )
    concept_review = _concept_review()
    source_refs = {str(row["source_ref"]) for row in migrated_rows}
    migration_payload = migration.to_dict()
    metrics = {
        "legacy_reference_unchanged": float(legacy_unchanged_after_migration),
        "legacy_reference_recall": float(legacy_recall) / float(max(1, len(migrated_rows))),
        "target_replay_recall": float(target_recall) / float(max(1, len(migrated_rows))),
        "source_isolation_ratio": float(len(source_refs)) / float(max(1, len(migrated_rows))),
        "migration_admission_ratio": float(migration_payload["admitted_count"]) / float(max(1, len(migrated_rows))),
        "incompatible_hold_count": len(migration.plan.held_entries),
        "concept_review_recovered": float(bool(concept_review.admission_plan.admitted_candidates)),
        "risa_reconstruction_observed": float(bool(risa_observations and kernel.state.graph.edges_by_key)),
        "migration_event_cost": sum(item.event_cost for item in migration.admissions),
        "target_state_budget_units": len(target_cache.entries),
    }
    passed = bool(
        metrics["legacy_reference_unchanged"] == 1.0
        and metrics["legacy_reference_recall"] == 1.0
        and metrics["target_replay_recall"] == 1.0
        and metrics["source_isolation_ratio"] == 1.0
        and metrics["migration_admission_ratio"] == 1.0
        and metrics["incompatible_hold_count"] == 1
        and metrics["concept_review_recovered"] == 1.0
        and metrics["risa_reconstruction_observed"] == 1.0
        and metrics["migration_event_cost"] <= 16
        and metrics["target_state_budget_units"] <= 6
    )
    return {
        "schema": "sara-architecture-migration-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "frozen_workload": True,
        "source_isolated_fixture": True,
        "independent_external_source_evidence": False,
        "workload_sha256": hashlib.sha256(workload_bytes).hexdigest(),
        "metrics": metrics,
        "migration": migration_payload,
        "retrieval_trace": retrieval_trace,
        "concept_review": concept_review.to_dict(),
        "risa_observation_count": len(risa_observations),
        "risa_graph_edge_count": len(kernel.state.graph.edges_by_key),
        "policy_notes": [
            "The fixture has isolated source references and revisions but is not independent external evidence.",
            "Legacy Event Memory is retained read-only while target replay is evaluated.",
            "Architecture promotion requires a separate independently sourced frozen workload.",
        ],
    }


def summarize(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    return "\n".join(
        (
            f"Architecture migration benchmark: {'PASS' if report['passed'] else 'FAIL'}",
            f"Frozen workload: {report['frozen_workload']}",
            f"Source-isolated fixture: {report['source_isolated_fixture']}",
            f"Independent external evidence: {report['independent_external_source_evidence']}",
            f"Legacy recall: {metrics['legacy_reference_recall']}",
            f"Target replay recall: {metrics['target_replay_recall']}",
            f"Concept review recovered: {metrics['concept_review_recovered']}",
            f"RISA reconstruction observed: {metrics['risa_reconstruction_observed']}",
        )
    ) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report()
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    summary_path = ensure_parent_directory(args.summary_path)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summarize(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
