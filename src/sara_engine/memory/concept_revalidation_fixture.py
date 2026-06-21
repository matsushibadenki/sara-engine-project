from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.ingest import make_candidate_relation


def _distinct_materials(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    distinct: List[Mapping[str, Any]] = []
    seen_refs = set()
    seen_hashes = set()
    for row in rows:
        source_ref = str(row.get("source_ref", "") or "")
        material_hash = str(row.get("material_hash", "") or "")
        if not source_ref or not material_hash:
            continue
        if source_ref in seen_refs or material_hash in seen_hashes:
            continue
        distinct.append(row)
        seen_refs.add(source_ref)
        seen_hashes.add(material_hash)
    return distinct


def _relation_dict(
    *,
    record_id: str,
    source_event_id: str,
    target_event_id: str,
    source_ref: str,
    source_hash: str,
    evidence_count: int,
    counterexample_count: int,
    prediction_gain: float,
) -> Dict[str, Any]:
    relation = make_candidate_relation(
        {
            "record_id": record_id,
            "relation": "predicts",
            "source_event_id": source_event_id,
            "target_event_id": target_event_id,
            "delay_lower_ms": 60,
            "delay_upper_ms": 140,
            "confidence": 0.88,
            "source_ref": source_ref,
            "source_hash": source_hash,
            "extractor_name": "prediction_gain",
            "extractor_version": "v1",
            "evidence_count": evidence_count,
            "counterexample_count": counterexample_count,
            "prediction_gain": prediction_gain,
        }
    )
    return relation.to_dict()


def build_concept_revalidation_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_cases: int = 4,
) -> List[Dict[str, Any]]:
    materials = _distinct_materials(rows)
    if len(materials) < 3:
        return []

    cases: List[Dict[str, Any]] = []

    first = materials[0]
    second = materials[1]
    third = materials[2]

    recoverable_key = "predicts:vision:visual_cluster_018->audio:audio_cluster_044"
    cases.append(
        {
            "schema": "sara-concept-revalidation-case-v1",
            "case_id": "recoverable_revision_conflict",
            "case_type": "recoverable_revision_conflict",
            "concept_key": recoverable_key,
            "expected_outcome": "admit",
            "queue_entry": {
                "concept_key": recoverable_key,
                "decision": "quarantine_source_revision_conflict",
                "supporting_relation_ids": [recoverable_key],
                "source_refs": [str(first.get("source_ref", ""))],
                "source_hashes": [str(first.get("material_hash", ""))],
                "revision_conflict_count": 1,
                "contradiction_score": 0.2,
                "next_action": "wait_for_source_revision_resolution",
                "attempt_count": 0,
                "blocked_at_segment": 1,
                "last_review_segment": 1,
                "retry_after_segment": 2,
            },
            "relations": [
                _relation_dict(
                    record_id="recoverable-a",
                    source_event_id="vision:visual_cluster_018",
                    target_event_id="audio:audio_cluster_044",
                    source_ref=str(first.get("source_ref", "")),
                    source_hash=str(first.get("material_hash", "")),
                    evidence_count=5,
                    counterexample_count=0,
                    prediction_gain=0.18,
                ),
                _relation_dict(
                    record_id="recoverable-b",
                    source_event_id="vision:visual_cluster_018",
                    target_event_id="audio:audio_cluster_044",
                    source_ref=str(second.get("source_ref", "")),
                    source_hash=str(second.get("material_hash", "")),
                    evidence_count=5,
                    counterexample_count=0,
                    prediction_gain=0.18,
                ),
            ],
        }
    )

    source_diversity_key = "predicts:vision:visual_cluster_019->audio:audio_cluster_045"
    cases.append(
        {
            "schema": "sara-concept-revalidation-case-v1",
            "case_id": "blocked_source_diversity",
            "case_type": "blocked_source_diversity",
            "concept_key": source_diversity_key,
            "expected_outcome": "blocked",
            "queue_entry": {
                "concept_key": source_diversity_key,
                "decision": "reject_insufficient_source_diversity",
                "supporting_relation_ids": [source_diversity_key],
                "source_refs": [str(first.get("source_ref", ""))],
                "source_hashes": [str(first.get("material_hash", ""))],
                "revision_conflict_count": 0,
                "contradiction_score": 0.0,
                "next_action": "collect_more_distinct_sources",
                "attempt_count": 0,
                "blocked_at_segment": 1,
                "last_review_segment": 1,
                "retry_after_segment": 2,
            },
            "relations": [
                _relation_dict(
                    record_id="source-diversity-a",
                    source_event_id="vision:visual_cluster_019",
                    target_event_id="audio:audio_cluster_045",
                    source_ref=str(first.get("source_ref", "")),
                    source_hash=str(first.get("material_hash", "")),
                    evidence_count=5,
                    counterexample_count=0,
                    prediction_gain=0.18,
                )
            ],
        }
    )

    counterexample_key = "predicts:vision:visual_cluster_020->audio:audio_cluster_046"
    cases.append(
        {
            "schema": "sara-concept-revalidation-case-v1",
            "case_id": "blocked_counterexample_pressure",
            "case_type": "blocked_counterexample_pressure",
            "concept_key": counterexample_key,
            "expected_outcome": "blocked",
            "queue_entry": {
                "concept_key": counterexample_key,
                "decision": "quarantine_counterexample_pressure",
                "supporting_relation_ids": [counterexample_key],
                "source_refs": [str(first.get("source_ref", "")), str(second.get("source_ref", ""))],
                "source_hashes": [str(first.get("material_hash", "")), str(second.get("material_hash", ""))],
                "revision_conflict_count": 0,
                "contradiction_score": 0.45,
                "next_action": "collect_counterexamples_and_retest",
                "attempt_count": 0,
                "blocked_at_segment": 1,
                "last_review_segment": 1,
                "retry_after_segment": 2,
            },
            "relations": [
                _relation_dict(
                    record_id="counterexample-a",
                    source_event_id="vision:visual_cluster_020",
                    target_event_id="audio:audio_cluster_046",
                    source_ref=str(first.get("source_ref", "")),
                    source_hash=str(first.get("material_hash", "")),
                    evidence_count=4,
                    counterexample_count=3,
                    prediction_gain=0.18,
                ),
                _relation_dict(
                    record_id="counterexample-b",
                    source_event_id="vision:visual_cluster_020",
                    target_event_id="audio:audio_cluster_046",
                    source_ref=str(second.get("source_ref", "")),
                    source_hash=str(second.get("material_hash", "")),
                    evidence_count=4,
                    counterexample_count=3,
                    prediction_gain=0.18,
                ),
            ],
        }
    )

    attempt_budget_key = "predicts:vision:visual_cluster_021->audio:audio_cluster_047"
    cases.append(
        {
            "schema": "sara-concept-revalidation-case-v1",
            "case_id": "blocked_attempt_budget",
            "case_type": "blocked_attempt_budget",
            "concept_key": attempt_budget_key,
            "expected_outcome": "blocked",
            "queue_entry": {
                "concept_key": attempt_budget_key,
                "decision": "reject_missing_support",
                "supporting_relation_ids": [attempt_budget_key],
                "source_refs": [str(third.get("source_ref", ""))],
                "source_hashes": [str(third.get("material_hash", ""))],
                "revision_conflict_count": 0,
                "contradiction_score": 0.0,
                "next_action": "rebuild_supporting_relations",
                "attempt_count": 3,
                "blocked_at_segment": 1,
                "last_review_segment": 1,
                "retry_after_segment": 2,
            },
            "relations": [],
        }
    )

    return cases[: max(1, int(max_cases))]


def summarize_case_types(cases: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for case in cases:
        key = str(case.get("case_type", "unknown") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))

