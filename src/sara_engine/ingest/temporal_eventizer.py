from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from .candidate_proposals import ObservedEvent, make_observed_event, make_proposal_lineage
from .change_detection import ChangePoint


@dataclass(frozen=True)
class EventizationTrace:
    emitted_count: int
    suppressed_count: int
    merge_window_ms: int
    schema: str = "sara-temporal-eventization-trace-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "emitted_count": int(self.emitted_count),
            "suppressed_count": int(self.suppressed_count),
            "merge_window_ms": int(self.merge_window_ms),
        }


class TemporalEventizer:
    """Groups nearby change points into bounded observed events."""

    def __init__(self, *, merge_window_ms: int = 120) -> None:
        self.merge_window_ms = max(0, int(merge_window_ms))
        self.last_trace = EventizationTrace(emitted_count=0, suppressed_count=0, merge_window_ms=self.merge_window_ms)

    def eventize(
        self,
        changes: Sequence[ChangePoint],
        *,
        source_ref: str,
        source_hash: str,
        extractor_version: str = "v1",
    ) -> List[ObservedEvent]:
        sorted_changes = sorted(changes, key=lambda item: (item.time_ms, item.stream_id))
        events: List[ObservedEvent] = []
        suppressed = 0
        current_group: List[ChangePoint] = []
        for change in sorted_changes:
            if not current_group:
                current_group = [change]
                continue
            if change.stream_id == current_group[-1].stream_id and (change.time_ms - current_group[-1].time_ms) <= self.merge_window_ms:
                current_group.append(change)
                continue
            events.append(self._group_to_event(current_group, source_ref, source_hash, extractor_version, len(events)))
            suppressed += max(0, len(current_group) - 1)
            current_group = [change]
        if current_group:
            events.append(self._group_to_event(current_group, source_ref, source_hash, extractor_version, len(events)))
            suppressed += max(0, len(current_group) - 1)
        self.last_trace = EventizationTrace(
            emitted_count=len(events),
            suppressed_count=suppressed,
            merge_window_ms=self.merge_window_ms,
        )
        return events

    def _group_to_event(
        self,
        changes: Sequence[ChangePoint],
        source_ref: str,
        source_hash: str,
        extractor_version: str,
        ordinal: int,
    ) -> ObservedEvent:
        first = changes[0]
        last = changes[-1]
        peak_delta = max(change.delta for change in changes)
        confidence = min(1.0, peak_delta / max(first.threshold, 1e-9))
        signature = [int(round(change.delta * 1000.0)) for change in changes[:32]]
        lineage = make_proposal_lineage(
            source_ref=source_ref,
            source_hash=source_hash,
            extractor_name="temporal_eventizer",
            extractor_version=extractor_version,
            parent_ids=[f"{change.stream_id}:{change.time_ms}" for change in changes],
            observed_anchor_ids=[f"{first.stream_id}:{first.time_ms}"],
        )
        return make_observed_event(
            {
                "record_id": f"{first.stream_id}-evt-{ordinal}",
                "modality": first.modality,
                "local_time_ms": first.time_ms,
                "duration_ms": max(0, last.time_ms - first.time_ms),
                "label": f"{first.modality}_change",
                "confidence": confidence,
                "sparse_signature": signature,
                "lineage": lineage,
            }
        )

