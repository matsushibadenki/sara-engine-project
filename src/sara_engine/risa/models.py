from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class ConceptCell:
    cell_id: str
    kind: str
    label: str
    attributes: Dict[str, str] = field(default_factory=dict)
    abstraction_level: int = 0
    created_at: int = 0
    usage_count: int = 0
    stability: float = 0.0
    recent_activity: float = 0.0
    energy: float = 0.5
    last_activated_at: int = 0
    dormant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConceptRelation:
    source: str
    target: str
    relation_type: str
    context_tags: Tuple[str, ...] = ()
    evidence_count: int = 0
    reliability: float = 0.0
    plasticity: float = 1.0
    last_updated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["context_tags"] = list(self.context_tags)
        return payload


@dataclass
class RisaObservation:
    timestamp: int
    actor: str
    action: str
    observed_effects: List[str]
    event_id: str = ""
    target: str | None = None
    context_tags: List[str] = field(default_factory=list)
    source_ref: str = ""
    verified: bool = True
    resonance_score: float = 0.0
    credit_longevity: float = 0.0
    event_energy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConceptPattern:
    pattern_id: str
    signature: str
    event_count: int = 0
    actors: set[str] = field(default_factory=set)
    actions: set[str] = field(default_factory=set)
    effects: set[str] = field(default_factory=set)
    support: int = 0
    context_tags: set[str] = field(default_factory=set)
    verified_support: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "signature": self.signature,
            "event_count": self.event_count,
            "actors": sorted(self.actors),
            "actions": sorted(self.actions),
            "effects": sorted(self.effects),
            "support": self.support,
            "context_tags": sorted(self.context_tags),
            "verified_support": self.verified_support,
        }


@dataclass
class RisaPredictionQuery:
    actor: str
    action: str
    target: str | None = None
    context_tags: List[str] = field(default_factory=list)


@dataclass
class RisaPredictionResult:
    predicted_effects: List[str]
    score: float
    supporting_paths: List[List[str]] = field(default_factory=list)
    evidence_event_ids: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
