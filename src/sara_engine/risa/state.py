from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .graph_store import RisaGraphStore
from .models import ConceptPattern, RisaObservation


@dataclass
class RisaKernelState:
    graph: RisaGraphStore = field(default_factory=RisaGraphStore)
    patterns: Dict[str, ConceptPattern] = field(default_factory=dict)
    observations_by_id: Dict[str, RisaObservation] = field(default_factory=dict)
    actor_action_effect_counts: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    action_effect_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    actor_action_context_effect_counts: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = field(
        default_factory=dict
    )
    action_context_effect_counts: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    concept_members: Dict[str, List[str]] = field(default_factory=dict)
    activation_index: Dict[str, List[str]] = field(default_factory=dict)
    concept_lineage: Dict[str, Dict[str, object]] = field(default_factory=dict)
    previous_observation_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "graph": self.graph.to_dict(),
            "patterns": {key: value.to_dict() for key, value in self.patterns.items()},
            "observations": {key: value.to_dict() for key, value in self.observations_by_id.items()},
            "actor_action_effect_counts": self.actor_action_effect_counts,
            "action_effect_counts": self.action_effect_counts,
            "actor_action_context_effect_counts": self.actor_action_context_effect_counts,
            "action_context_effect_counts": self.action_context_effect_counts,
            "concept_members": self.concept_members,
            "activation_index": self.activation_index,
            "concept_lineage": self.concept_lineage,
            "previous_observation_id": self.previous_observation_id,
        }
