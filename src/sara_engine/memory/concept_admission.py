from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Sequence, Tuple

from sara_engine.ingest import (
    CandidateRelation,
    ConceptAuditResult,
    ConceptCrystallizationGuard,
    ConceptCrystalCandidate,
    FrequentSequence,
)
from sara_engine.memory.event_state_cache import EventStateCandidate


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stable_id(text: str, modulus: int = 4096) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % max(1, int(modulus))


@dataclass(frozen=True)
class ConceptRevalidationEntry:
    concept_key: str
    decision: str
    supporting_relation_ids: Tuple[str, ...]
    source_refs: Tuple[str, ...]
    source_hashes: Tuple[str, ...]
    revision_conflict_count: int
    contradiction_score: float
    next_action: str
    attempt_count: int = 0
    blocked_at_segment: int = 0
    last_review_segment: int = 0
    retry_after_segment: int = 0
    schema: str = "sara-concept-revalidation-entry-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "concept_key": self.concept_key,
            "decision": self.decision,
            "supporting_relation_ids": list(self.supporting_relation_ids),
            "source_refs": list(self.source_refs),
            "source_hashes": list(self.source_hashes),
            "revision_conflict_count": int(self.revision_conflict_count),
            "contradiction_score": float(self.contradiction_score),
            "next_action": self.next_action,
            "attempt_count": int(self.attempt_count),
            "blocked_at_segment": int(self.blocked_at_segment),
            "last_review_segment": int(self.last_review_segment),
            "retry_after_segment": int(self.retry_after_segment),
        }


@dataclass(frozen=True)
class ConceptAdmissionPlan:
    admitted_candidates: Tuple[EventStateCandidate, ...]
    revalidation_queue: Tuple[ConceptRevalidationEntry, ...]
    audits: Tuple[ConceptAuditResult, ...]
    schema: str = "sara-concept-admission-plan-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "admitted_candidates": [item.entry_id for item in self.admitted_candidates],
            "revalidation_queue": [item.to_dict() for item in self.revalidation_queue],
            "audits": [item.to_dict() for item in self.audits],
        }


class ConceptAdmissionPlanner:
    """Bridges audited concept candidates into Event Memory admission candidates."""

    def __init__(
        self,
        *,
        guard: ConceptCrystallizationGuard | None = None,
        signature_width: int = 8,
        default_metabolic_headroom: float = 1.0,
    ) -> None:
        self.guard = guard or ConceptCrystallizationGuard()
        self.signature_width = max(2, int(signature_width))
        self.default_metabolic_headroom = _clamp01(default_metabolic_headroom)

    def build_plan(
        self,
        relations: Sequence[CandidateRelation],
        concept_candidates: Sequence[ConceptCrystalCandidate],
        *,
        time_segment: int,
        frequent_sequences: Sequence[FrequentSequence] = (),
    ) -> ConceptAdmissionPlan:
        audits = self.guard.audit_candidates(
            relations,
            concept_candidates,
            frequent_sequences=frequent_sequences,
        )
        admitted: List[EventStateCandidate] = []
        revalidation: List[ConceptRevalidationEntry] = []
        for candidate, audit in zip(concept_candidates, audits):
            supporting_relations = self._supporting_relations(relations, candidate.concept_key)
            if audit.accepted:
                admitted.append(
                    self._to_event_state_candidate(
                        candidate,
                        audit,
                        supporting_relations,
                        time_segment=time_segment,
                    )
                )
            else:
                revalidation.append(
                    self._to_revalidation_entry(candidate, audit, supporting_relations, time_segment=time_segment)
                )
        return ConceptAdmissionPlan(
            admitted_candidates=tuple(admitted),
            revalidation_queue=tuple(revalidation),
            audits=tuple(audits),
        )

    def _to_event_state_candidate(
        self,
        candidate: ConceptCrystalCandidate,
        audit: ConceptAuditResult,
        supporting_relations: Sequence[CandidateRelation],
        *,
        time_segment: int,
    ) -> EventStateCandidate:
        source_refs = sorted(
            {item.lineage.source_ref for item in supporting_relations if item.lineage.source_ref}
        )
        source_hashes = sorted(
            {item.lineage.source_hash for item in supporting_relations if item.lineage.source_hash}
        )
        signature = self._signature_for_candidate(candidate, supporting_relations)
        source_ref = source_refs[0] if len(source_refs) == 1 else f"concept::{candidate.concept_key}"
        source_revision = self._aggregate_revision(source_hashes)
        source_reliability = min(1.0, 0.5 * candidate.confidence + 0.5 * (audit.distinct_source_hashes / max(1, audit.distinct_source_refs)))
        return EventStateCandidate(
            entry_id=candidate.record_id,
            signature=signature,
            source_ref=source_ref,
            source_revision=source_revision,
            time_segment=int(time_segment),
            own_latent_id=candidate.concept_key,
            causal_predecessors=tuple(candidate.supporting_relation_ids),
            confidence=_clamp01(candidate.confidence),
            uncertainty=round(1.0 - _clamp01(candidate.confidence), 6),
            source_reliability=round(_clamp01(source_reliability), 6),
            resonance_score=_clamp01(candidate.confidence),
            sequence_support_score=_clamp01(audit.sequence_support_score),
            sequence_support_count=max(0, int(audit.sequence_support_count)),
            metabolic_headroom=self.default_metabolic_headroom,
            observed=True,
            source_backed=bool(audit.distinct_source_hashes > 0),
            verified=True,
            contradicted=False,
            abstained=False,
            event_cost=max(1, int(candidate.evidence_count) + len(signature)),
        )

    def _to_revalidation_entry(
        self,
        candidate: ConceptCrystalCandidate,
        audit: ConceptAuditResult,
        supporting_relations: Sequence[CandidateRelation],
        *,
        time_segment: int,
    ) -> ConceptRevalidationEntry:
        source_refs = tuple(
            sorted({item.lineage.source_ref for item in supporting_relations if item.lineage.source_ref})
        )
        source_hashes = tuple(
            sorted({item.lineage.source_hash for item in supporting_relations if item.lineage.source_hash})
        )
        next_action = "collect_more_distinct_sources"
        if audit.decision == "quarantine_source_revision_conflict":
            next_action = "wait_for_source_revision_resolution"
        elif audit.decision == "quarantine_counterexample_pressure":
            next_action = "collect_counterexamples_and_retest"
        elif audit.decision == "reject_missing_support":
            next_action = "rebuild_supporting_relations"
        return ConceptRevalidationEntry(
            concept_key=candidate.concept_key,
            decision=audit.decision,
            supporting_relation_ids=tuple(candidate.supporting_relation_ids),
            source_refs=source_refs,
            source_hashes=source_hashes,
            revision_conflict_count=audit.revision_conflict_count,
            contradiction_score=audit.contradiction_score,
            next_action=next_action,
            attempt_count=0,
            blocked_at_segment=int(time_segment),
            last_review_segment=int(time_segment),
            retry_after_segment=int(time_segment) + 1,
        )

    def _supporting_relations(
        self,
        relations: Sequence[CandidateRelation],
        concept_key: str,
    ) -> Tuple[CandidateRelation, ...]:
        return tuple(
            relation
            for relation in relations
            if f"{relation.relation}:{relation.source_event_id}->{relation.target_event_id}" == concept_key
        )

    def _signature_for_candidate(
        self,
        candidate: ConceptCrystalCandidate,
        supporting_relations: Sequence[CandidateRelation],
    ) -> Tuple[int, ...]:
        pieces: List[int] = [_stable_id(candidate.concept_key)]
        for relation_id in candidate.supporting_relation_ids:
            pieces.append(_stable_id(f"support:{relation_id}"))
        for relation in supporting_relations:
            pieces.append(_stable_id(f"source:{relation.lineage.source_ref}"))
            pieces.append(_stable_id(f"hash:{relation.lineage.source_hash}"))
        ordered = sorted(set(pieces))
        return tuple(ordered[: self.signature_width])

    def _aggregate_revision(self, source_hashes: Sequence[str]) -> str:
        if not source_hashes:
            return ""
        digest = hashlib.sha256("|".join(sorted(source_hashes)).encode("utf-8")).hexdigest()
        return f"concept-rev:{digest[:16]}"
