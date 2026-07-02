from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.dynamics import concept_self_state_alignment
from sara_engine.learning.adaptive_credit import summarize_event_memory_credit
from sara_engine.ingest import CandidateRelation, ConceptCrystallizationGuard, FrequentSequence, summarize_sequence_relation_support
from sara_engine.memory.concept_admission import ConceptRevalidationEntry


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class ConceptRetryDecision:
    concept_key: str
    ready: bool
    decision: str
    next_action: str
    priority_score: float
    matched_relation_count: int
    sequence_support_count: int
    sequence_support_score: float
    credit_score: float
    credit_confidence: float
    credit_longevity: float
    multimodal_bundle_affinity: float
    self_state_alignment_score: float
    distinct_source_refs: int
    distinct_source_hashes: int
    revision_conflict_count: int
    contradiction_score: float
    retry_after_segment: int
    attempt_count: int
    schema: str = "sara-concept-retry-decision-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "concept_key": self.concept_key,
            "ready": bool(self.ready),
            "decision": self.decision,
            "next_action": self.next_action,
            "priority_score": float(self.priority_score),
            "matched_relation_count": int(self.matched_relation_count),
            "sequence_support_count": int(self.sequence_support_count),
            "sequence_support_score": float(self.sequence_support_score),
            "credit_score": float(self.credit_score),
            "credit_confidence": float(self.credit_confidence),
            "credit_longevity": float(self.credit_longevity),
            "multimodal_bundle_affinity": float(self.multimodal_bundle_affinity),
            "self_state_alignment_score": float(self.self_state_alignment_score),
            "distinct_source_refs": int(self.distinct_source_refs),
            "distinct_source_hashes": int(self.distinct_source_hashes),
            "revision_conflict_count": int(self.revision_conflict_count),
            "contradiction_score": float(self.contradiction_score),
            "retry_after_segment": int(self.retry_after_segment),
            "attempt_count": int(self.attempt_count),
        }


@dataclass(frozen=True)
class ConceptRevalidationSchedule:
    ready_queue: Tuple[ConceptRetryDecision, ...]
    blocked_queue: Tuple[ConceptRetryDecision, ...]
    schema: str = "sara-concept-revalidation-schedule-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "ready_queue": [item.to_dict() for item in self.ready_queue],
            "blocked_queue": [item.to_dict() for item in self.blocked_queue],
        }


class ConceptRevalidationScheduler:
    """Determines when quarantined concept candidates are ready for reassessment."""

    def __init__(
        self,
        *,
        guard: ConceptCrystallizationGuard | None = None,
        min_cooldown_segments: int = 1,
        max_attempts: int = 3,
    ) -> None:
        self.guard = guard or ConceptCrystallizationGuard()
        self.min_cooldown_segments = max(0, int(min_cooldown_segments))
        self.max_attempts = max(1, int(max_attempts))

    def build_schedule(
        self,
        queue_entries: Sequence[ConceptRevalidationEntry],
        relations: Sequence[CandidateRelation],
        *,
        current_segment: int,
        frequent_sequences: Sequence[FrequentSequence] = (),
        self_state_ids: Sequence[int] = (),
    ) -> ConceptRevalidationSchedule:
        sequence_support = summarize_sequence_relation_support(relations, frequent_sequences)
        ready: List[ConceptRetryDecision] = []
        blocked: List[ConceptRetryDecision] = []
        for entry in queue_entries:
            decision = self._decision_for_entry(
                entry,
                relations,
                current_segment=current_segment,
                sequence_support=sequence_support.get(entry.concept_key),
                self_state_ids=self_state_ids,
            )
            if decision.ready:
                ready.append(decision)
            else:
                blocked.append(decision)
        ready.sort(key=lambda item: (-item.priority_score, item.concept_key))
        blocked.sort(key=lambda item: (item.retry_after_segment, item.concept_key))
        return ConceptRevalidationSchedule(
            ready_queue=tuple(ready),
            blocked_queue=tuple(blocked),
        )

    def mark_dispatched(
        self,
        entry: ConceptRevalidationEntry,
        *,
        current_segment: int,
    ) -> ConceptRevalidationEntry:
        return ConceptRevalidationEntry(
            concept_key=entry.concept_key,
            decision=entry.decision,
            supporting_relation_ids=entry.supporting_relation_ids,
            source_refs=entry.source_refs,
            source_hashes=entry.source_hashes,
            revision_conflict_count=entry.revision_conflict_count,
            contradiction_score=entry.contradiction_score,
            next_action=entry.next_action,
            attempt_count=entry.attempt_count + 1,
            blocked_at_segment=entry.blocked_at_segment,
            last_review_segment=int(current_segment),
            retry_after_segment=int(current_segment) + self.min_cooldown_segments,
        )

    def _decision_for_entry(
        self,
        entry: ConceptRevalidationEntry,
        relations: Sequence[CandidateRelation],
        *,
        current_segment: int,
        sequence_support: Any = None,
        self_state_ids: Sequence[int] = (),
    ) -> ConceptRetryDecision:
        stats = self._stats_for_concept(entry.concept_key, relations)
        self_state_alignment = concept_self_state_alignment(entry.concept_key, self_state_ids)
        credit_summary = self._credit_summary_for_relations(stats["matched_relations"])
        if entry.attempt_count >= self.max_attempts:
            return self._make_decision(
                entry,
                ready=False,
                decision="blocked_attempt_budget",
                next_action="manual_review",
                stats=stats,
                retry_after_segment=max(entry.retry_after_segment, current_segment),
                priority_score=0.0,
                self_state_alignment=self_state_alignment,
                credit_summary=credit_summary,
            )
        if current_segment < max(entry.retry_after_segment, entry.last_review_segment + self.min_cooldown_segments):
            return self._make_decision(
                entry,
                ready=False,
                decision="blocked_cooldown",
                next_action=entry.next_action,
                stats=stats,
                retry_after_segment=max(entry.retry_after_segment, entry.last_review_segment + self.min_cooldown_segments),
                priority_score=0.0,
                self_state_alignment=self_state_alignment,
                credit_summary=credit_summary,
            )

        ready = False
        decision = "hold_missing_requirements"
        next_action = entry.next_action
        if entry.decision == "quarantine_source_revision_conflict":
            if stats["revision_conflict_count"] == 0 and stats["distinct_source_refs"] >= self.guard.min_distinct_source_refs:
                ready = True
                decision = "ready_revision_conflict_resolved"
                next_action = "re_audit_concept_candidate"
        elif entry.decision == "quarantine_counterexample_pressure":
            if (
                stats["contradiction_score"] <= self.guard.max_counterexample_rate
                and stats["revision_conflict_count"] == 0
                and stats["distinct_source_refs"] >= self.guard.min_distinct_source_refs
            ):
                ready = True
                decision = "ready_counterexample_pressure_reduced"
                next_action = "re_audit_concept_candidate"
        elif entry.decision == "reject_insufficient_source_diversity":
            if stats["distinct_source_refs"] >= self.guard.min_distinct_source_refs:
                ready = True
                decision = "ready_source_diversity_recovered"
                next_action = "re_audit_concept_candidate"
        elif entry.decision == "reject_missing_support":
            if stats["matched_relation_count"] > 0:
                ready = True
                decision = "ready_support_rebuilt"
                next_action = "rebuild_concept_candidate"

        priority = 0.0
        if ready:
            priority = round(
                0.40 * min(1.0, stats["distinct_source_refs"] / max(1, self.guard.min_distinct_source_refs))
                + 0.20 * min(1.0, stats["matched_relation_count"] / 4.0)
                + 0.15 * (float(getattr(sequence_support, "support_score", 0.0) or 0.0))
                + 0.10 * min(1.0, float(getattr(sequence_support, "supporting_sequence_count", 0) or 0) / 3.0)
                + 0.05 * _clamp01(self_state_alignment)
                + 0.05 * float(credit_summary.get("credit_score", 0.0) or 0.0)
                + 0.03 * float(credit_summary.get("multimodal_bundle_affinity", 0.0) or 0.0)
                + 0.05 * (1.0 - _clamp01(stats["contradiction_score"])),
                6,
            )

        return self._make_decision(
            entry,
            ready=ready,
            decision=decision,
            next_action=next_action,
            stats=stats,
            retry_after_segment=max(entry.retry_after_segment, current_segment + (0 if ready else self.min_cooldown_segments)),
            priority_score=priority,
            sequence_support=sequence_support,
            self_state_alignment=self_state_alignment,
            credit_summary=credit_summary,
        )

    def _stats_for_concept(
        self,
        concept_key: str,
        relations: Sequence[CandidateRelation],
    ) -> Mapping[str, Any]:
        matched = [
            relation
            for relation in relations
            if f"{relation.relation}:{relation.source_event_id}->{relation.target_event_id}" == concept_key
        ]
        source_refs = {
            relation.lineage.source_ref for relation in matched if relation.lineage.source_ref
        }
        source_hashes = {
            relation.lineage.source_hash for relation in matched if relation.lineage.source_hash
        }
        source_ref_hashes: Dict[str, set[str]] = {}
        evidence_count = 0
        counterexample_count = 0
        for relation in matched:
            evidence_count += max(0, int(relation.evidence_count))
            counterexample_count += max(0, int(relation.counterexample_count))
            source_ref = str(relation.lineage.source_ref)
            source_hash = str(relation.lineage.source_hash)
            if source_ref and source_hash:
                source_ref_hashes.setdefault(source_ref, set()).add(source_hash)
        revision_conflict_count = sum(1 for hashes in source_ref_hashes.values() if len(hashes) > 1)
        contradiction_score = float(counterexample_count) / float(max(1, evidence_count + counterexample_count))
        return {
            "matched_relation_count": len(matched),
            "matched_relations": tuple(matched),
            "distinct_source_refs": len(source_refs),
            "distinct_source_hashes": len(source_hashes),
            "revision_conflict_count": revision_conflict_count,
            "contradiction_score": contradiction_score,
        }

    def _make_decision(
        self,
        entry: ConceptRevalidationEntry,
        *,
        ready: bool,
        decision: str,
        next_action: str,
        stats: Mapping[str, Any],
        retry_after_segment: int,
        priority_score: float,
        sequence_support: Any = None,
        self_state_alignment: float = 0.0,
        credit_summary: Mapping[str, Any] | None = None,
    ) -> ConceptRetryDecision:
        credit_summary = credit_summary or {}
        return ConceptRetryDecision(
            concept_key=entry.concept_key,
            ready=ready,
            decision=decision,
            next_action=next_action,
            priority_score=priority_score,
            matched_relation_count=int(stats["matched_relation_count"]),
            sequence_support_count=int(getattr(sequence_support, "supporting_sequence_count", 0) or 0),
            sequence_support_score=float(getattr(sequence_support, "support_score", 0.0) or 0.0),
            credit_score=float(credit_summary.get("credit_score", 0.0) or 0.0),
            credit_confidence=float(credit_summary.get("credit_confidence", 0.0) or 0.0),
            credit_longevity=float(credit_summary.get("credit_longevity", 0.0) or 0.0),
            multimodal_bundle_affinity=float(credit_summary.get("multimodal_bundle_affinity", 0.0) or 0.0),
            self_state_alignment_score=float(_clamp01(self_state_alignment)),
            distinct_source_refs=int(stats["distinct_source_refs"]),
            distinct_source_hashes=int(stats["distinct_source_hashes"]),
            revision_conflict_count=int(stats["revision_conflict_count"]),
            contradiction_score=float(stats["contradiction_score"]),
            retry_after_segment=int(retry_after_segment),
            attempt_count=int(entry.attempt_count),
        )

    def _credit_summary_for_relations(
        self,
        relations: Sequence[CandidateRelation],
    ) -> Dict[str, float]:
        route_states: List[Dict[str, float]] = []
        for relation in relations:
            evidence = max(1.0, float(max(0, int(relation.evidence_count))))
            counterexamples = float(max(0, int(relation.counterexample_count)))
            bundle_affinity = 0.0
            if str(relation.lineage.source_ref or "").startswith("bundle::"):
                bundle_affinity = 0.8
            if (
                str(relation.source_event_id or "").startswith("bundle:")
                or str(relation.target_event_id or "").startswith("bundle:")
            ):
                bundle_affinity = 1.0
            route_states.append(
                {
                    "responsibility": _clamp01(float(relation.prediction_gain) * min(1.0, evidence / 4.0)),
                    "confidence": _clamp01(float(relation.confidence)),
                    "longevity": _clamp01(evidence / (evidence + counterexamples + 1.0)),
                    "multimodal_bundle_affinity": bundle_affinity,
                }
            )
        summary = summarize_event_memory_credit(route_states)
        summary["multimodal_bundle_affinity"] = round(
            max(
                [float(state.get("multimodal_bundle_affinity", 0.0) or 0.0) for state in route_states]
                + [0.0]
            ),
            6,
        )
        return summary
