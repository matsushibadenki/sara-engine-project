from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .candidate_proposals import CandidateRelation
from .frequent_sequence import FrequentSequence


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _relation_key(relation: CandidateRelation) -> str:
    return f"{relation.relation}:{relation.source_event_id}->{relation.target_event_id}"


def _label_from_event_id(event_id: str) -> str:
    parts = str(event_id).split(":", 1)
    return parts[1] if len(parts) == 2 else str(event_id)


@dataclass(frozen=True)
class SequenceRelationSupport:
    concept_key: str
    supporting_sequence_count: int
    supporting_episode_count: int
    ordered_match_count: int
    mean_span_ms: float
    support_score: float
    schema: str = "sara-sequence-relation-support-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "concept_key": self.concept_key,
            "supporting_sequence_count": int(self.supporting_sequence_count),
            "supporting_episode_count": int(self.supporting_episode_count),
            "ordered_match_count": int(self.ordered_match_count),
            "mean_span_ms": float(self.mean_span_ms),
            "support_score": float(self.support_score),
        }


def summarize_sequence_relation_support(
    relations: Sequence[CandidateRelation],
    sequences: Sequence[FrequentSequence],
) -> Dict[str, SequenceRelationSupport]:
    summaries: Dict[str, SequenceRelationSupport] = {}
    for relation in relations:
        concept_key = _relation_key(relation)
        source_label = _label_from_event_id(relation.source_event_id)
        target_label = _label_from_event_id(relation.target_event_id)
        matching_sequences: List[FrequentSequence] = []
        ordered_match_count = 0
        for sequence in sequences:
            try:
                source_index = sequence.labels.index(source_label)
                target_index = sequence.labels.index(target_label)
            except ValueError:
                continue
            matching_sequences.append(sequence)
            if source_index < target_index:
                ordered_match_count += 1
        if not matching_sequences:
            summaries[concept_key] = SequenceRelationSupport(
                concept_key=concept_key,
                supporting_sequence_count=0,
                supporting_episode_count=0,
                ordered_match_count=0,
                mean_span_ms=0.0,
                support_score=0.0,
            )
            continue
        episode_count = len(
            {
                episode_id
                for sequence in matching_sequences
                for episode_id in sequence.parent_episode_ids
            }
        )
        mean_span_ms = float(sum(sequence.mean_span_ms for sequence in matching_sequences)) / float(
            len(matching_sequences)
        )
        order_ratio = float(ordered_match_count) / float(len(matching_sequences))
        episode_factor = min(1.0, float(episode_count) / 4.0)
        sequence_factor = min(1.0, float(len(matching_sequences)) / 3.0)
        compact_span_factor = 1.0 / (1.0 + max(0.0, mean_span_ms) / 250.0)
        support_score = _clamp01(
            0.45 * order_ratio
            + 0.30 * episode_factor
            + 0.15 * sequence_factor
            + 0.10 * compact_span_factor
        )
        summaries[concept_key] = SequenceRelationSupport(
            concept_key=concept_key,
            supporting_sequence_count=len(matching_sequences),
            supporting_episode_count=episode_count,
            ordered_match_count=ordered_match_count,
            mean_span_ms=mean_span_ms,
            support_score=support_score,
        )
    return summaries
