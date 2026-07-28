from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple


IR_VERSION = "sara-canonical-ir-v1"
STATE_SCHEMA = "sara-canonical-ir-state-v1"
DEFAULT_MAX_EVENTS = 10_000
MAX_TEXT_LENGTH = 256
MAX_TAGS_PER_EVENT = 32
_REQUIRED_EVENT_FIELDS = frozenset(
    {"event_id", "timestep", "channel", "spike_id", "modality"}
)
_OPTIONAL_EVENT_FIELDS = frozenset({"confidence", "tags"})
_EVENT_FIELDS = _REQUIRED_EVENT_FIELDS | _OPTIONAL_EVENT_FIELDS


def _validated_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if len(value) > MAX_TEXT_LENGTH:
        raise ValueError(f"{field} exceeds {MAX_TEXT_LENGTH} characters")
    return value


def _validated_index(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _validated_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be a finite number")
    confidence = float(value)
    if not math.isfinite(confidence):
        raise ValueError("confidence must be a finite number")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    rounded = round(confidence, 6)
    return 0.0 if rounded == 0.0 else rounded


def _validated_tags(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("tags must be a list or tuple")
    if len(value) > MAX_TAGS_PER_EVENT:
        raise ValueError(f"tags exceeds {MAX_TAGS_PER_EVENT} entries")
    return tuple(
        sorted(
            {
                _validated_text(tag, field="tag")
                for tag in value
            }
        )
    )


@dataclass(frozen=True)
class CanonicalSparseEvent:
    event_id: str
    timestep: int
    channel: str
    spike_id: int
    modality: str
    confidence: float = 1.0
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "event_id", _validated_text(self.event_id, field="event_id")
        )
        object.__setattr__(
            self, "timestep", _validated_index(self.timestep, field="timestep")
        )
        object.__setattr__(
            self, "channel", _validated_text(self.channel, field="channel")
        )
        object.__setattr__(
            self, "spike_id", _validated_index(self.spike_id, field="spike_id")
        )
        object.__setattr__(
            self, "modality", _validated_text(self.modality, field="modality")
        )
        object.__setattr__(
            self, "confidence", _validated_confidence(self.confidence)
        )
        object.__setattr__(self, "tags", _validated_tags(self.tags))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestep": self.timestep,
            "channel": self.channel,
            "spike_id": self.spike_id,
            "modality": self.modality,
            "confidence": self.confidence,
            "tags": list(self.tags),
        }


def canonicalize_events(
    events: Iterable[Mapping[str, Any]],
    *,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> Tuple[CanonicalSparseEvent, ...]:
    if (
        isinstance(max_events, bool)
        or not isinstance(max_events, int)
        or max_events < 1
    ):
        raise ValueError("max_events must be a positive integer")
    normalized = []
    event_ids = set()
    for index, event in enumerate(events):
        if index >= max_events:
            raise ValueError(f"event count exceeds max_events={max_events}")
        if not isinstance(event, Mapping):
            raise ValueError(f"event at index {index} must be a mapping")
        missing = sorted(_REQUIRED_EVENT_FIELDS.difference(event))
        if missing:
            raise ValueError(
                f"event at index {index} is missing fields: {', '.join(missing)}"
            )
        unknown = sorted(set(event).difference(_EVENT_FIELDS))
        if unknown:
            raise ValueError(
                f"event at index {index} has unknown fields: {', '.join(unknown)}"
            )
        normalized_event = CanonicalSparseEvent(
            event_id=event["event_id"],
            timestep=event["timestep"],
            channel=event["channel"],
            spike_id=event["spike_id"],
            modality=event["modality"],
            confidence=event.get("confidence", 1.0),
            tags=event.get("tags", []),
        )
        if normalized_event.event_id in event_ids:
            raise ValueError(
                f"duplicate event_id: {normalized_event.event_id}"
            )
        event_ids.add(normalized_event.event_id)
        normalized.append(normalized_event)
    return tuple(
        sorted(
            normalized,
            key=lambda event: (
                event.timestep,
                event.event_id,
                event.spike_id,
                event.channel,
                event.modality,
                event.confidence,
                event.tags,
            ),
        )
    )


def canonical_json(
    events: Iterable[Mapping[str, Any]],
    *,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> str:
    payload = [
        event.to_dict()
        for event in canonicalize_events(events, max_events=max_events)
    ]
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def replay_digest(
    events: Iterable[Mapping[str, Any]],
    *,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> str:
    return hashlib.sha256(
        canonical_json(events, max_events=max_events).encode("utf-8")
    ).hexdigest()


def migrate_state(
    state: Mapping[str, Any],
    *,
    from_version: str,
    to_version: str,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> Dict[str, Any]:
    if from_version != IR_VERSION or to_version != IR_VERSION:
        raise ValueError("unsupported canonical sparse IR migration")
    if not isinstance(state, Mapping):
        raise ValueError("canonical state must be a mapping")
    unknown = sorted(set(state).difference({"schema", "ir_version", "events"}))
    if unknown:
        raise ValueError(
            f"canonical state has unknown fields: {', '.join(unknown)}"
        )
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("unsupported canonical sparse IR state schema")
    if state.get("ir_version") != from_version:
        raise ValueError("canonical state ir_version does not match from_version")
    events = state.get("events")
    if not isinstance(events, list):
        raise ValueError("canonical state events must be a list")
    return {
        "schema": STATE_SCHEMA,
        "ir_version": to_version,
        "events": [
            event.to_dict()
            for event in canonicalize_events(events, max_events=max_events)
        ],
    }
