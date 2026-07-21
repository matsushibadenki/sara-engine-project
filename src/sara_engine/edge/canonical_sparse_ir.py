from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple


@dataclass(frozen=True)
class CanonicalSparseEvent:
    event_id: str
    timestep: int
    channel: str
    spike_id: int
    modality: str
    confidence: float = 1.0
    tags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestep": int(self.timestep),
            "channel": self.channel,
            "spike_id": int(self.spike_id),
            "modality": self.modality,
            "confidence": round(float(self.confidence), 6),
            "tags": list(sorted(set(self.tags))),
        }


def canonicalize_events(events: Iterable[Mapping[str, Any]]) -> Tuple[CanonicalSparseEvent, ...]:
    normalized = tuple(
        CanonicalSparseEvent(
            event_id=str(event["event_id"]),
            timestep=int(event["timestep"]),
            channel=str(event["channel"]),
            spike_id=int(event["spike_id"]),
            modality=str(event["modality"]),
            confidence=float(event.get("confidence", 1.0)),
            tags=tuple(str(tag) for tag in event.get("tags", [])),
        )
        for event in events
    )
    return tuple(sorted(normalized, key=lambda event: (event.timestep, event.event_id, event.spike_id)))


def replay_digest(events: Iterable[Mapping[str, Any]]) -> str:
    payload = [event.to_dict() for event in canonicalize_events(events)]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def migrate_state(state: Mapping[str, Any], *, from_version: str, to_version: str) -> Dict[str, Any]:
    if from_version != "sara-canonical-ir-v1" or to_version != "sara-canonical-ir-v1":
        raise ValueError("unsupported canonical sparse IR migration")
    if state.get("schema") != "sara-canonical-ir-state-v1":
        raise ValueError("unsupported canonical sparse IR state schema")
    events = state.get("events")
    if not isinstance(events, list):
        raise ValueError("canonical state events must be a list")
    return {
        "schema": "sara-canonical-ir-state-v1",
        "ir_version": to_version,
        "events": [event.to_dict() for event in canonicalize_events(events)],
    }
