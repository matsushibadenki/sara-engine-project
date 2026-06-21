from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union

from .candidate_proposals import CandidateEvent, CandidateRelation, ObservedEvent, make_candidate_relation

EventSurface = Union[ObservedEvent, CandidateEvent]


@dataclass(frozen=True)
class PredictionGainTrace:
    pair_count: int
    accepted_pair_count: int
    min_support: int
    min_gain: float
    schema: str = "sara-prediction-gain-trace-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "pair_count": int(self.pair_count),
            "accepted_pair_count": int(self.accepted_pair_count),
            "min_support": int(self.min_support),
            "min_gain": float(self.min_gain),
        }


class PredictionGainEstimator:
    """Computes bounded next-event gain and emits candidate relations."""

    def __init__(self, *, min_support: int = 2, min_gain: float = 0.05, max_delay_ms: int = 400) -> None:
        self.min_support = max(1, int(min_support))
        self.min_gain = float(min_gain)
        self.max_delay_ms = max(1, int(max_delay_ms))
        self.last_trace = PredictionGainTrace(pair_count=0, accepted_pair_count=0, min_support=self.min_support, min_gain=self.min_gain)

    def propose_relations(self, events: Sequence[EventSurface]) -> List[CandidateRelation]:
        ordered = sorted(events, key=lambda item: (item.local_time_ms, item.record_id))
        antecedent_counts: Dict[str, int] = {}
        consequent_counts: Dict[str, int] = {}
        pair_counts: Dict[tuple[str, str], int] = {}
        delay_samples: Dict[tuple[str, str], List[int]] = {}
        representative_events: Dict[str, EventSurface] = {}

        for event in ordered:
            event_key = self._event_key(event)
            antecedent_counts[event_key] = antecedent_counts.get(event_key, 0) + 1
            consequent_counts[event_key] = consequent_counts.get(event_key, 0) + 1
            representative_events.setdefault(event_key, event)

        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                delay = right.local_time_ms - left.local_time_ms
                if delay <= 0 or delay > self.max_delay_ms:
                    break
                pair_key = (self._event_key(left), self._event_key(right))
                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
                delay_samples.setdefault(pair_key, []).append(delay)

        accepted: List[CandidateRelation] = []
        for pair_key, support in sorted(pair_counts.items()):
            left_id, right_id = pair_key
            if support < self.min_support:
                continue
            p_b = float(consequent_counts.get(right_id, 0)) / float(max(1, len(ordered)))
            p_b_given_a = float(support) / float(max(1, antecedent_counts.get(left_id, 1)))
            gain = p_b_given_a - p_b
            if gain < self.min_gain:
                continue
            left_event = representative_events[left_id]
            right_event = representative_events[right_id]
            delays = delay_samples[pair_key]
            accepted.append(
                make_candidate_relation(
                    {
                        "record_id": f"{left_id}__predicts__{right_id}",
                        "relation": "predicts",
                        "source_event_id": left_id,
                        "target_event_id": right_id,
                        "delay_lower_ms": min(delays),
                        "delay_upper_ms": max(delays),
                        "confidence": min(1.0, p_b_given_a),
                        "source_ref": left_event.lineage.source_ref,
                        "source_hash": left_event.lineage.source_hash,
                        "extractor_name": "prediction_gain",
                        "extractor_version": "v1",
                        "parent_ids": [left_id, right_id],
                        "observed_anchor_ids": [left_id, right_id],
                        "evidence_count": support,
                        "counterexample_count": max(0, antecedent_counts.get(left_id, 0) - support),
                        "prediction_gain": gain,
                    }
                )
            )
        self.last_trace = PredictionGainTrace(
            pair_count=len(pair_counts),
            accepted_pair_count=len(accepted),
            min_support=self.min_support,
            min_gain=self.min_gain,
        )
        return accepted

    def _event_key(self, event: EventSurface) -> str:
        return f"{event.modality}:{event.label}"
