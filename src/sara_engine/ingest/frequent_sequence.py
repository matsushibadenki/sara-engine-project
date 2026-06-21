from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from .candidate_proposals import CandidateEvent, ObservedEvent
from .episode_segmentation import BoundedEpisode


@dataclass(frozen=True)
class FrequentSequence:
    sequence_key: str
    labels: Tuple[str, ...]
    support_episode_count: int
    occurrence_count: int
    source_count: int
    mean_span_ms: float
    parent_episode_ids: Tuple[str, ...]
    schema: str = "sara-frequent-sequence-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "sequence_key": self.sequence_key,
            "labels": list(self.labels),
            "support_episode_count": int(self.support_episode_count),
            "occurrence_count": int(self.occurrence_count),
            "source_count": int(self.source_count),
            "mean_span_ms": float(self.mean_span_ms),
            "parent_episode_ids": list(self.parent_episode_ids),
        }


@dataclass(frozen=True)
class FrequentSequenceTrace:
    considered_sequences: int
    accepted_sequences: int
    min_support_episodes: int
    max_pattern_length: int
    max_span_ms: int
    schema: str = "sara-frequent-sequence-trace-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "considered_sequences": int(self.considered_sequences),
            "accepted_sequences": int(self.accepted_sequences),
            "min_support_episodes": int(self.min_support_episodes),
            "max_pattern_length": int(self.max_pattern_length),
            "max_span_ms": int(self.max_span_ms),
        }


class FrequentSequenceMiner:
    """Finds repeated bounded event subsequences across episodes."""

    def __init__(
        self,
        *,
        min_support_episodes: int = 2,
        max_pattern_length: int = 3,
        max_span_ms: int = 400,
    ) -> None:
        self.min_support_episodes = max(1, int(min_support_episodes))
        self.max_pattern_length = max(2, int(max_pattern_length))
        self.max_span_ms = max(1, int(max_span_ms))
        self.last_trace = FrequentSequenceTrace(
            considered_sequences=0,
            accepted_sequences=0,
            min_support_episodes=self.min_support_episodes,
            max_pattern_length=self.max_pattern_length,
            max_span_ms=self.max_span_ms,
        )

    def mine(
        self,
        episodes: Sequence[BoundedEpisode],
        observed_events: Sequence[ObservedEvent],
        *,
        candidate_events: Sequence[CandidateEvent] = (),
    ) -> List[FrequentSequence]:
        event_index = self._build_event_index(observed_events, candidate_events)
        buckets: MutableMapping[Tuple[str, ...], Dict[str, Any]] = {}
        considered = 0
        for episode in episodes:
            ordered_items = self._episode_items(episode, event_index)
            for start in range(len(ordered_items)):
                for length in range(2, self.max_pattern_length + 1):
                    end = start + length
                    if end > len(ordered_items):
                        break
                    segment = ordered_items[start:end]
                    span_ms = segment[-1]["time_ms"] - segment[0]["time_ms"]
                    if span_ms > self.max_span_ms:
                        break
                    considered += 1
                    labels = tuple(item["label"] for item in segment)
                    bucket = buckets.setdefault(
                        labels,
                        {
                            "occurrence_count": 0,
                            "episode_ids": set(),
                            "source_refs": set(),
                            "spans": [],
                        },
                    )
                    bucket["occurrence_count"] += 1
                    bucket["episode_ids"].add(episode.episode_id)
                    if episode.source_ref:
                        bucket["source_refs"].add(episode.source_ref)
                    bucket["spans"].append(span_ms)

        accepted: List[FrequentSequence] = []
        for labels, bucket in sorted(buckets.items()):
            support_episode_count = len(bucket["episode_ids"])
            if support_episode_count < self.min_support_episodes:
                continue
            spans = bucket["spans"]
            mean_span_ms = (float(sum(spans)) / float(len(spans))) if spans else 0.0
            accepted.append(
                FrequentSequence(
                    sequence_key=" -> ".join(labels),
                    labels=labels,
                    support_episode_count=support_episode_count,
                    occurrence_count=int(bucket["occurrence_count"]),
                    source_count=len(bucket["source_refs"]),
                    mean_span_ms=mean_span_ms,
                    parent_episode_ids=tuple(sorted(bucket["episode_ids"])),
                )
            )

        self.last_trace = FrequentSequenceTrace(
            considered_sequences=considered,
            accepted_sequences=len(accepted),
            min_support_episodes=self.min_support_episodes,
            max_pattern_length=self.max_pattern_length,
            max_span_ms=self.max_span_ms,
        )
        return accepted

    def _build_event_index(
        self,
        observed_events: Sequence[ObservedEvent],
        candidate_events: Sequence[CandidateEvent],
    ) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for event in observed_events:
            index[event.record_id] = {
                "label": event.label,
                "time_ms": int(event.local_time_ms),
                "record_type": event.record_type,
            }
        for event in candidate_events:
            index[event.record_id] = {
                "label": event.label,
                "time_ms": int(event.local_time_ms),
                "record_type": event.record_type,
            }
        return index

    def _episode_items(
        self,
        episode: BoundedEpisode,
        event_index: Mapping[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for record_id in episode.parent_ids:
            payload = event_index.get(record_id)
            if payload is None:
                continue
            items.append(dict(payload))
        items.sort(key=lambda item: (int(item["time_ms"]), str(item["label"])))
        return items
