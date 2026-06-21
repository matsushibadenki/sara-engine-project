from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union

from .candidate_proposals import CandidateEvent, CandidateRelation, ObservedEvent, make_candidate_relation

EventSurface = Union[ObservedEvent, CandidateEvent]


@dataclass(frozen=True)
class SynchronyTrace:
    compared_pairs: int
    accepted_pairs: int
    window_ms: int
    cross_modal_only: bool
    schema: str = "sara-synchrony-trace-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "compared_pairs": int(self.compared_pairs),
            "accepted_pairs": int(self.accepted_pairs),
            "window_ms": int(self.window_ms),
            "cross_modal_only": bool(self.cross_modal_only),
        }


class SynchronyDetector:
    """Detects bounded near-time synchrony across observed event streams."""

    def __init__(self, *, window_ms: int = 80, cross_modal_only: bool = True) -> None:
        self.window_ms = max(1, int(window_ms))
        self.cross_modal_only = bool(cross_modal_only)
        self.last_trace = SynchronyTrace(
            compared_pairs=0,
            accepted_pairs=0,
            window_ms=self.window_ms,
            cross_modal_only=self.cross_modal_only,
        )

    def propose_relations(self, events: Sequence[EventSurface]) -> List[CandidateRelation]:
        ordered = sorted(events, key=lambda item: (item.local_time_ms, item.record_id))
        relations: List[CandidateRelation] = []
        compared_pairs = 0
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                delay = right.local_time_ms - left.local_time_ms
                if delay > self.window_ms:
                    break
                compared_pairs += 1
                if self.cross_modal_only and left.modality == right.modality:
                    continue
                confidence = self._confidence_from_delay(delay)
                relations.append(
                    make_candidate_relation(
                        {
                            "record_id": f"{left.record_id}__synchronized_with__{right.record_id}",
                            "relation": "synchronized_with",
                            "source_event_id": left.record_id,
                            "target_event_id": right.record_id,
                            "delay_lower_ms": min(0, delay),
                            "delay_upper_ms": delay,
                            "confidence": confidence,
                            "source_ref": left.lineage.source_ref or right.lineage.source_ref,
                            "source_hash": left.lineage.source_hash or right.lineage.source_hash,
                            "extractor_name": "synchrony_detector",
                            "extractor_version": "v1",
                            "parent_ids": [left.record_id, right.record_id],
                            "observed_anchor_ids": [left.record_id, right.record_id],
                            "evidence_count": 1,
                            "counterexample_count": 0,
                            "prediction_gain": 0.0,
                        }
                    )
                )
        self.last_trace = SynchronyTrace(
            compared_pairs=compared_pairs,
            accepted_pairs=len(relations),
            window_ms=self.window_ms,
            cross_modal_only=self.cross_modal_only,
        )
        return relations

    def _confidence_from_delay(self, delay_ms: int) -> float:
        normalized = max(0.0, 1.0 - (float(delay_ms) / float(self.window_ms)))
        return max(0.05, normalized)
