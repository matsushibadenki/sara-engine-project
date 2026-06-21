from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .candidate_proposals import CandidateRelation, ConceptCrystalCandidate
from .frequent_sequence import FrequentSequence
from .sequence_relation_support import summarize_sequence_relation_support


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class ConceptAuditResult:
    concept_key: str
    accepted: bool
    decision: str
    supporting_relation_count: int
    distinct_source_refs: int
    distinct_source_hashes: int
    revision_conflict_count: int
    contradiction_score: float
    sequence_support_score: float
    sequence_support_count: int
    trace: Dict[str, Any]
    schema: str = "sara-concept-audit-result-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "concept_key": self.concept_key,
            "accepted": bool(self.accepted),
            "decision": self.decision,
            "supporting_relation_count": int(self.supporting_relation_count),
            "distinct_source_refs": int(self.distinct_source_refs),
            "distinct_source_hashes": int(self.distinct_source_hashes),
            "revision_conflict_count": int(self.revision_conflict_count),
            "contradiction_score": float(self.contradiction_score),
            "sequence_support_score": float(self.sequence_support_score),
            "sequence_support_count": int(self.sequence_support_count),
            "trace": dict(self.trace),
        }


class ConceptCrystallizationGuard:
    """Prevents fragile concept candidates from entering durable memory."""

    def __init__(
        self,
        *,
        min_distinct_source_refs: int = 2,
        max_counterexample_rate: float = 0.25,
        max_revision_conflicts: int = 0,
        min_sequence_support_score: float = 0.0,
    ) -> None:
        self.min_distinct_source_refs = max(1, int(min_distinct_source_refs))
        self.max_counterexample_rate = _clamp01(max_counterexample_rate)
        self.max_revision_conflicts = max(0, int(max_revision_conflicts))
        self.min_sequence_support_score = _clamp01(min_sequence_support_score)

    def audit_candidates(
        self,
        relations: Sequence[CandidateRelation],
        concept_candidates: Sequence[ConceptCrystalCandidate],
        *,
        frequent_sequences: Sequence[FrequentSequence] = (),
    ) -> List[ConceptAuditResult]:
        relation_map = self._group_relations(relations)
        sequence_support = summarize_sequence_relation_support(relations, frequent_sequences)
        return [
            self._audit_single(
                candidate,
                relation_map.get(candidate.concept_key, ()),
                sequence_support_score=float(sequence_support.get(candidate.concept_key).support_score if candidate.concept_key in sequence_support else 0.0),
                sequence_support_count=int(sequence_support.get(candidate.concept_key).supporting_sequence_count if candidate.concept_key in sequence_support else 0),
            )
            for candidate in concept_candidates
        ]

    def accepted_candidates(
        self,
        relations: Sequence[CandidateRelation],
        concept_candidates: Sequence[ConceptCrystalCandidate],
        *,
        frequent_sequences: Sequence[FrequentSequence] = (),
    ) -> List[ConceptCrystalCandidate]:
        relation_map = self._group_relations(relations)
        sequence_support = summarize_sequence_relation_support(relations, frequent_sequences)
        accepted: List[ConceptCrystalCandidate] = []
        for candidate in concept_candidates:
            audit = self._audit_single(
                candidate,
                relation_map.get(candidate.concept_key, ()),
                sequence_support_score=float(sequence_support.get(candidate.concept_key).support_score if candidate.concept_key in sequence_support else 0.0),
                sequence_support_count=int(sequence_support.get(candidate.concept_key).supporting_sequence_count if candidate.concept_key in sequence_support else 0),
            )
            if audit.accepted:
                accepted.append(candidate)
        return accepted

    def _audit_single(
        self,
        candidate: ConceptCrystalCandidate,
        supporting_relations: Sequence[CandidateRelation],
        *,
        sequence_support_score: float = 0.0,
        sequence_support_count: int = 0,
    ) -> ConceptAuditResult:
        if not supporting_relations:
            return ConceptAuditResult(
                concept_key=candidate.concept_key,
                accepted=False,
                decision="reject_missing_support",
                supporting_relation_count=0,
                distinct_source_refs=0,
                distinct_source_hashes=0,
                revision_conflict_count=0,
                contradiction_score=1.0,
                sequence_support_score=0.0,
                sequence_support_count=0,
                trace={"thresholds": self._threshold_trace()},
            )

        source_refs = {
            relation.lineage.source_ref for relation in supporting_relations if relation.lineage.source_ref
        }
        source_hashes = {
            relation.lineage.source_hash for relation in supporting_relations if relation.lineage.source_hash
        }
        revision_conflicts = self._revision_conflicts(supporting_relations)
        evidence_count = sum(max(0, int(item.evidence_count)) for item in supporting_relations)
        counterexample_count = sum(max(0, int(item.counterexample_count)) for item in supporting_relations)
        contradiction_score = float(counterexample_count) / float(max(1, evidence_count + counterexample_count))

        if len(source_refs) < self.min_distinct_source_refs:
            decision = "reject_insufficient_source_diversity"
            accepted = False
        elif len(revision_conflicts) > self.max_revision_conflicts:
            decision = "quarantine_source_revision_conflict"
            accepted = False
        elif contradiction_score > self.max_counterexample_rate:
            decision = "quarantine_counterexample_pressure"
            accepted = False
        elif sequence_support_score < self.min_sequence_support_score:
            decision = "reject_weak_sequence_support"
            accepted = False
        else:
            decision = "accept_concept_candidate"
            accepted = True

        trace = {
            "source_refs": sorted(source_refs),
            "source_hashes": sorted(source_hashes),
            "revision_conflict_refs": sorted(revision_conflicts),
            "evidence_count": evidence_count,
            "counterexample_count": counterexample_count,
            "sequence_support_score": sequence_support_score,
            "sequence_support_count": sequence_support_count,
            "thresholds": self._threshold_trace(),
        }
        return ConceptAuditResult(
            concept_key=candidate.concept_key,
            accepted=accepted,
            decision=decision,
            supporting_relation_count=len(supporting_relations),
            distinct_source_refs=len(source_refs),
            distinct_source_hashes=len(source_hashes),
            revision_conflict_count=len(revision_conflicts),
            contradiction_score=contradiction_score,
            sequence_support_score=sequence_support_score,
            sequence_support_count=sequence_support_count,
            trace=trace,
        )

    def _group_relations(
        self,
        relations: Sequence[CandidateRelation],
    ) -> Dict[str, Tuple[CandidateRelation, ...]]:
        grouped: Dict[str, List[CandidateRelation]] = {}
        for relation in relations:
            key = f"{relation.relation}:{relation.source_event_id}->{relation.target_event_id}"
            grouped.setdefault(key, []).append(relation)
        return {key: tuple(values) for key, values in grouped.items()}

    def _revision_conflicts(
        self,
        supporting_relations: Sequence[CandidateRelation],
    ) -> Tuple[str, ...]:
        source_ref_hashes: Dict[str, set[str]] = {}
        for relation in supporting_relations:
            source_ref = str(relation.lineage.source_ref)
            source_hash = str(relation.lineage.source_hash)
            if not source_ref or not source_hash:
                continue
            source_ref_hashes.setdefault(source_ref, set()).add(source_hash)
        conflicts = [source_ref for source_ref, hashes in source_ref_hashes.items() if len(hashes) > 1]
        return tuple(sorted(conflicts))

    def _threshold_trace(self) -> Mapping[str, Any]:
        return {
            "min_distinct_source_refs": self.min_distinct_source_refs,
            "max_counterexample_rate": self.max_counterexample_rate,
            "max_revision_conflicts": self.max_revision_conflicts,
            "min_sequence_support_score": self.min_sequence_support_score,
        }
