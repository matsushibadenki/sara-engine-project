from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .candidate_proposals import CandidateRelation, ConceptCrystalCandidate, make_proposal_lineage


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


@dataclass(frozen=True)
class StableRelationSummary:
    relation_key: str
    relation: str
    source_event_id: str
    target_event_id: str
    context_count: int
    source_count: int
    evidence_count: int
    counterexample_count: int
    mean_prediction_gain: float
    min_prediction_gain: float
    stability_score: float
    schema: str = "sara-stable-relation-summary-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "relation_key": self.relation_key,
            "relation": self.relation,
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "context_count": int(self.context_count),
            "source_count": int(self.source_count),
            "evidence_count": int(self.evidence_count),
            "counterexample_count": int(self.counterexample_count),
            "mean_prediction_gain": float(self.mean_prediction_gain),
            "min_prediction_gain": float(self.min_prediction_gain),
            "stability_score": float(self.stability_score),
        }


class RelationStabilityAssessor:
    """Scores whether relations remain useful across multiple contexts and sources."""

    def __init__(
        self,
        *,
        min_contexts: int = 2,
        min_stability_score: float = 0.35,
        min_mean_prediction_gain: float = 0.05,
    ) -> None:
        self.min_contexts = max(1, int(min_contexts))
        self.min_stability_score = _clamp01(min_stability_score)
        self.min_mean_prediction_gain = float(min_mean_prediction_gain)

    def summarize(self, relations: Sequence[CandidateRelation]) -> List[StableRelationSummary]:
        buckets: Dict[str, List[CandidateRelation]] = {}
        for relation in relations:
            buckets.setdefault(self._relation_key(relation), []).append(relation)

        summaries: List[StableRelationSummary] = []
        for key, bucket in sorted(buckets.items()):
            contexts = {relation.lineage.source_ref for relation in bucket if relation.lineage.source_ref}
            sources = {relation.lineage.source_hash for relation in bucket if relation.lineage.source_hash}
            evidence_total = sum(max(0, int(item.evidence_count)) for item in bucket)
            counterexample_total = sum(max(0, int(item.counterexample_count)) for item in bucket)
            gains = [float(item.prediction_gain) for item in bucket]
            mean_gain = _mean(gains)
            min_gain = min(gains) if gains else 0.0
            counterexample_rate = float(counterexample_total) / float(max(1, evidence_total + counterexample_total))
            source_factor = min(1.0, float(len(sources)) / float(max(1, self.min_contexts)))
            context_factor = min(1.0, float(len(contexts)) / float(max(1, self.min_contexts)))
            gain_factor = max(0.0, mean_gain)
            stability_score = _clamp01((0.35 * context_factor) + (0.25 * source_factor) + (0.40 * gain_factor) - (0.30 * counterexample_rate))
            exemplar = bucket[0]
            summaries.append(
                StableRelationSummary(
                    relation_key=key,
                    relation=exemplar.relation,
                    source_event_id=exemplar.source_event_id,
                    target_event_id=exemplar.target_event_id,
                    context_count=len(contexts),
                    source_count=len(sources),
                    evidence_count=evidence_total,
                    counterexample_count=counterexample_total,
                    mean_prediction_gain=mean_gain,
                    min_prediction_gain=min_gain,
                    stability_score=stability_score,
                )
            )
        return summaries

    def crystallization_candidates(self, relations: Sequence[CandidateRelation]) -> List[ConceptCrystalCandidate]:
        candidates: List[ConceptCrystalCandidate] = []
        for summary in self.summarize(relations):
            if summary.context_count < self.min_contexts:
                continue
            if summary.mean_prediction_gain < self.min_mean_prediction_gain:
                continue
            if summary.stability_score < self.min_stability_score:
                continue
            lineage = make_proposal_lineage(
                source_ref="multi_context",
                source_hash="multi_source",
                extractor_name="relation_stability",
                extractor_version="v1",
                parent_ids=[summary.relation_key],
                observed_anchor_ids=[summary.source_event_id, summary.target_event_id],
            )
            candidates.append(
                ConceptCrystalCandidate(
                    record_id=f"concept::{summary.relation_key}",
                    concept_key=summary.relation_key,
                    supporting_relation_ids=(summary.relation_key,),
                    confidence=summary.stability_score,
                    evidence_count=summary.evidence_count,
                    counterexample_count=summary.counterexample_count,
                    prediction_gain=summary.mean_prediction_gain,
                    lineage=lineage,
                )
            )
        return candidates

    def _relation_key(self, relation: CandidateRelation) -> str:
        return f"{relation.relation}:{relation.source_event_id}->{relation.target_event_id}"

