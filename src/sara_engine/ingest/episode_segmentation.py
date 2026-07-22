from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, List, Sequence, Tuple

from .candidate_proposals import CandidateEvent, ObservedEvent


@dataclass(frozen=True)
class BoundedEpisode:
    episode_id: str
    source_ref: str
    source_hash: str
    start_time_ms: int
    end_time_ms: int
    observed_event_ids: Tuple[str, ...]
    candidate_event_ids: Tuple[str, ...]
    parent_ids: Tuple[str, ...]
    modalities: Tuple[str, ...]
    event_count: int
    schema: str = "sara-bounded-episode-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "episode_id": self.episode_id,
            "source_ref": self.source_ref,
            "source_hash": self.source_hash,
            "start_time_ms": int(self.start_time_ms),
            "end_time_ms": int(self.end_time_ms),
            "observed_event_ids": list(self.observed_event_ids),
            "candidate_event_ids": list(self.candidate_event_ids),
            "parent_ids": list(self.parent_ids),
            "modalities": list(self.modalities),
            "event_count": int(self.event_count),
        }


@dataclass(frozen=True)
class EpisodeSegmentationTrace:
    episode_count: int
    overflow_split_count: int
    source_split_count: int
    gap_split_count: int
    max_gap_ms: int
    max_events_per_episode: int
    schema: str = "sara-episode-segmentation-trace-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "episode_count": int(self.episode_count),
            "overflow_split_count": int(self.overflow_split_count),
            "source_split_count": int(self.source_split_count),
            "gap_split_count": int(self.gap_split_count),
            "max_gap_ms": int(self.max_gap_ms),
            "max_events_per_episode": int(self.max_events_per_episode),
        }


@dataclass(frozen=True)
class MultimodalEpisodeBridgeResult:
    episode: BoundedEpisode | None
    connected: bool
    reason: str
    event_cost: int
    durable_mutation_allowed: bool = False
    schema: str = "sara-multimodal-episode-bridge-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "episode": self.episode.to_dict() if self.episode is not None else None,
            "connected": bool(self.connected),
            "reason": self.reason,
            "event_cost": int(self.event_cost),
            "durable_mutation_allowed": False,
        }


def bridge_verified_bundle_to_episode(
    bundle: Any,
    admission: Any,
    *,
    max_events_per_episode: int = 12,
) -> MultimodalEpisodeBridgeResult:
    """Project one admitted bundle into a bounded Event Memory episode."""
    limit = max(1, int(max_events_per_episode))
    child_records = tuple(getattr(bundle, "child_records", ()) or ())
    candidate = getattr(admission, "candidate", None)
    event_cost = len(child_records)
    receipt = getattr(candidate, "verification_receipt", None)
    receipt_valid = bool(
        receipt is not None
        and callable(getattr(receipt, "is_valid", None))
        and receipt.is_valid()
        and getattr(receipt, "verifier_id", "")
        == "multimodal-event-memory-admission"
        and getattr(receipt, "verified", False)
        and getattr(receipt, "decision", "")
        == str(getattr(admission, "promotion_decision", ""))
        and getattr(receipt, "source_revision", "")
        == str(getattr(candidate, "source_revision", ""))
        and str(getattr(candidate, "source_ref", ""))
        in set(getattr(receipt, "source_refs", ()) or ())
    )
    verified_boundary = bool(
        candidate is not None
        and getattr(admission, "promotion_allowed", False)
        and getattr(candidate, "verified", False)
        and getattr(candidate, "observed", False)
        and getattr(candidate, "source_backed", False)
        and str(getattr(candidate, "entry_id", ""))
        == str(getattr(bundle, "event_id", ""))
        and receipt_valid
    )
    if not verified_boundary:
        return MultimodalEpisodeBridgeResult(
            episode=None,
            connected=False,
            reason="bundle_not_verified_for_episode",
            event_cost=event_cost,
        )
    if not child_records or len(child_records) > limit:
        return MultimodalEpisodeBridgeResult(
            episode=None,
            connected=False,
            reason="episode_event_budget_exceeded",
            event_cost=event_cost,
        )
    if not all(
        getattr(item, "observed", False)
        and str(getattr(item, "event_id", "")).strip()
        and str(getattr(item, "source_ref", "")).strip()
        for item in child_records
    ):
        return MultimodalEpisodeBridgeResult(
            episode=None,
            connected=False,
            reason="episode_evidence_incomplete",
            event_cost=event_cost,
        )
    modalities = tuple(
        sorted({str(getattr(item, "modality", "")) for item in child_records})
    )
    if len(modalities) < 2:
        return MultimodalEpisodeBridgeResult(
            episode=None,
            connected=False,
            reason="episode_requires_multiple_modalities",
            event_cost=event_cost,
        )
    ordered = tuple(
        sorted(
            child_records,
            key=lambda item: (
                float(getattr(item, "timestamp_ms", 0.0)),
                str(getattr(item, "event_id", "")),
            ),
        )
    )
    source_rows = tuple(
        sorted(
            f"{getattr(item, 'modality', '')}|{getattr(item, 'source_ref', '')}|"
            f"{getattr(item, 'event_id', '')}"
            for item in ordered
        )
    )
    source_hash = sha256("\n".join(source_rows).encode("utf-8")).hexdigest()
    episode = BoundedEpisode(
        episode_id=f"multimodal::{getattr(bundle, 'event_id', '')}",
        source_ref=str(getattr(candidate, "source_ref", "")),
        source_hash=source_hash,
        start_time_ms=int(min(float(getattr(item, "timestamp_ms", 0.0)) for item in ordered)),
        end_time_ms=int(max(float(getattr(item, "timestamp_ms", 0.0)) for item in ordered)),
        observed_event_ids=tuple(str(getattr(item, "event_id", "")) for item in ordered),
        candidate_event_ids=(),
        parent_ids=tuple(str(getattr(item, "event_id", "")) for item in ordered),
        modalities=modalities,
        event_count=len(ordered),
        schema="sara-bounded-multimodal-episode-v1",
    )
    return MultimodalEpisodeBridgeResult(
        episode=episode,
        connected=True,
        reason="verified_bundle_episode_connected",
        event_cost=event_cost,
    )


@dataclass(frozen=True)
class _EpisodeItem:
    record_id: str
    modality: str
    local_time_ms: int
    source_ref: str
    source_hash: str
    record_type: str


class EpisodeSegmenter:
    """Builds bounded episodes shared by observed and proposal-assisted lanes."""

    def __init__(self, *, max_gap_ms: int = 250, max_events_per_episode: int = 12) -> None:
        self.max_gap_ms = max(1, int(max_gap_ms))
        self.max_events_per_episode = max(1, int(max_events_per_episode))
        self.last_trace = EpisodeSegmentationTrace(
            episode_count=0,
            overflow_split_count=0,
            source_split_count=0,
            gap_split_count=0,
            max_gap_ms=self.max_gap_ms,
            max_events_per_episode=self.max_events_per_episode,
        )

    def segment(
        self,
        observed_events: Sequence[ObservedEvent],
        *,
        candidate_events: Sequence[CandidateEvent] = (),
    ) -> List[BoundedEpisode]:
        items = self._normalize_items(observed_events, candidate_events)
        episodes: List[BoundedEpisode] = []
        current: List[_EpisodeItem] = []
        overflow_split_count = 0
        source_split_count = 0
        gap_split_count = 0
        for item in items:
            if not current:
                current = [item]
                continue
            split_reason = self._split_reason(current, item)
            if split_reason:
                episodes.append(self._build_episode(current, len(episodes)))
                if split_reason == "overflow":
                    overflow_split_count += 1
                elif split_reason == "source":
                    source_split_count += 1
                elif split_reason == "gap":
                    gap_split_count += 1
                current = [item]
                continue
            current.append(item)
        if current:
            episodes.append(self._build_episode(current, len(episodes)))

        self.last_trace = EpisodeSegmentationTrace(
            episode_count=len(episodes),
            overflow_split_count=overflow_split_count,
            source_split_count=source_split_count,
            gap_split_count=gap_split_count,
            max_gap_ms=self.max_gap_ms,
            max_events_per_episode=self.max_events_per_episode,
        )
        return episodes

    def _normalize_items(
        self,
        observed_events: Sequence[ObservedEvent],
        candidate_events: Sequence[CandidateEvent],
    ) -> List[_EpisodeItem]:
        items: List[_EpisodeItem] = []
        for event in observed_events:
            items.append(
                _EpisodeItem(
                    record_id=event.record_id,
                    modality=event.modality,
                    local_time_ms=int(event.local_time_ms),
                    source_ref=event.lineage.source_ref,
                    source_hash=event.lineage.source_hash,
                    record_type=event.record_type,
                )
            )
        for event in candidate_events:
            items.append(
                _EpisodeItem(
                    record_id=event.record_id,
                    modality=event.modality,
                    local_time_ms=int(event.local_time_ms),
                    source_ref=event.lineage.source_ref,
                    source_hash=event.lineage.source_hash,
                    record_type=event.record_type,
                )
            )
        return sorted(items, key=lambda item: (item.source_ref, item.source_hash, item.local_time_ms, item.record_id))

    def _split_reason(
        self,
        current: Sequence[_EpisodeItem],
        next_item: _EpisodeItem,
    ) -> str:
        last = current[-1]
        if len(current) >= self.max_events_per_episode:
            return "overflow"
        if (last.source_ref, last.source_hash) != (next_item.source_ref, next_item.source_hash):
            return "source"
        if (next_item.local_time_ms - last.local_time_ms) > self.max_gap_ms:
            return "gap"
        return ""

    def _build_episode(self, items: Sequence[_EpisodeItem], ordinal: int) -> BoundedEpisode:
        first = items[0]
        last = items[-1]
        observed_ids = tuple(item.record_id for item in items if item.record_type == "observed_event")
        candidate_ids = tuple(item.record_id for item in items if item.record_type == "candidate_event")
        parent_ids = tuple(item.record_id for item in items)
        modalities = tuple(sorted({item.modality for item in items}))
        return BoundedEpisode(
            episode_id=f"{first.source_ref or 'source'}::episode::{ordinal}",
            source_ref=first.source_ref,
            source_hash=first.source_hash,
            start_time_ms=first.local_time_ms,
            end_time_ms=last.local_time_ms,
            observed_event_ids=observed_ids,
            candidate_event_ids=candidate_ids,
            parent_ids=parent_ids,
            modalities=modalities,
            event_count=len(items),
        )
