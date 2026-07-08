from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from .models import ConceptCell, ConceptPattern, ConceptRelation, RisaObservation, RisaPredictionQuery, RisaPredictionResult
from .state import RisaKernelState


def _normalize_label(value: str) -> str:
    return "_".join(str(value or "").strip().lower().split())


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _context_key(context_tags: Sequence[str]) -> str:
    normalized = sorted(_normalize_label(tag) for tag in context_tags if str(tag).strip())
    return "|".join(normalized) or "__no_context__"


def _index_append(index: Dict[str, List[str]], key: str, value: str) -> None:
    values = index.setdefault(key, [])
    if value not in values:
        values.append(value)


@dataclass(frozen=True)
class RisaKernelSnapshot:
    state: Dict[str, object]


class SARAAlignedRisaKernel:
    """Concept-cell kernel ported from RISA and aligned with SARA event semantics."""

    def __init__(
        self,
        *,
        min_support: int = 2,
        min_distinct_actors: int = 2,
        dormancy_energy_threshold: float = 0.12,
        dormancy_idle_threshold: int = 50,
        connection_cost_rate: float = 0.015,
    ) -> None:
        self.state = RisaKernelState()
        self.min_support = max(1, int(min_support))
        self.min_distinct_actors = max(1, int(min_distinct_actors))
        self.dormancy_energy_threshold = _clamp01(dormancy_energy_threshold)
        self.dormancy_idle_threshold = max(1, int(dormancy_idle_threshold))
        self.connection_cost_rate = max(0.0, float(connection_cost_rate))

    def ingest_observation(self, observation: RisaObservation) -> None:
        event_id = observation.event_id or f"obs:{len(self.state.observations_by_id) + 1}"
        event = RisaObservation(
            timestamp=int(observation.timestamp),
            actor=str(observation.actor),
            action=str(observation.action),
            observed_effects=[str(item) for item in observation.observed_effects],
            event_id=str(event_id),
            target=observation.target,
            context_tags=[str(item) for item in observation.context_tags],
            source_ref=str(observation.source_ref),
            verified=bool(observation.verified),
            resonance_score=_clamp01(observation.resonance_score),
            credit_longevity=_clamp01(observation.credit_longevity),
            event_energy=_clamp01(observation.event_energy),
        )
        self.state.observations_by_id[event.event_id] = event
        self._build_graph_links(event)
        if event.verified:
            self._learn_verified_pattern(event)
            self._rebuild_concepts(event.timestamp)
        self.apply_metabolism(event.timestamp)
        self.state.previous_observation_id = event.event_id

    def ingest_observations(self, observations: Iterable[RisaObservation]) -> None:
        for observation in observations:
            self.ingest_observation(observation)

    def predict(self, query: RisaPredictionQuery) -> RisaPredictionResult:
        actor = _normalize_label(query.actor)
        action = _normalize_label(query.action)
        context_key = _context_key(query.context_tags)

        actor_scores = self.state.actor_action_effect_counts.get(actor, {}).get(action, {})
        action_scores = self.state.action_effect_counts.get(action, {})
        actor_context_scores = (
            self.state.actor_action_context_effect_counts.get(actor, {}).get(action, {}).get(context_key, {})
        )
        action_context_scores = self.state.action_context_effect_counts.get(action, {}).get(context_key, {})
        candidate_effects = self._collect_candidates(actor, action, context_key)
        if not candidate_effects:
            return RisaPredictionResult(
                predicted_effects=[],
                score=0.0,
                explanation="No matching verified concept or event pattern found.",
            )

        best_effect = ""
        best_score = -1.0
        for effect in candidate_effects:
            direct_total = sum(actor_scores.values())
            action_total = sum(action_scores.values())
            actor_context_total = sum(actor_context_scores.values())
            action_context_total = sum(action_context_scores.values())
            direct_match_score = (actor_scores.get(effect, 0) / direct_total) if direct_total else 0.0
            action_pattern_score = (action_scores.get(effect, 0) / action_total) if action_total else 0.0
            actor_context_score = (actor_context_scores.get(effect, 0) / actor_context_total) if actor_context_total else 0.0
            action_context_score = (action_context_scores.get(effect, 0) / action_context_total) if action_context_total else 0.0

            concept_support = 0.0
            concept_id = f"concept:shared_{action}_{effect}"
            if concept_id in self.state.concept_members:
                members = self.state.concept_members[concept_id]
                if actor in members:
                    concept_support = 1.0
                elif members:
                    concept_support = 0.6

            score = (
                (0.25 * direct_match_score)
                + (0.25 * action_pattern_score)
                + (0.20 * actor_context_score)
                + (0.15 * action_context_score)
                + (0.15 * concept_support)
            )
            if score > best_score:
                best_score = score
                best_effect = effect

        supporting_paths = [[f"entity:{actor}", f"process:{action}", f"state:{best_effect}"]]
        concept_id = f"concept:shared_{action}_{best_effect}"
        if concept_id in self.state.concept_members:
            supporting_paths.append([f"entity:{actor}", concept_id, f"state:{best_effect}"])

        evidence_event_ids = [
            event.event_id
            for event in self.state.observations_by_id.values()
            if event.verified
            if _normalize_label(event.action) == action
            and best_effect in [_normalize_label(item) for item in event.observed_effects]
        ]
        explanation = (
            f"Predicted {best_effect} from verified action-pattern, context-pattern, and concept-cell support."
        )
        return RisaPredictionResult(
            predicted_effects=[best_effect],
            score=round(best_score, 4),
            supporting_paths=supporting_paths,
            evidence_event_ids=sorted(evidence_event_ids),
            explanation=explanation,
        )

    def apply_metabolism(self, current_timestamp: int) -> None:
        for node in self.state.graph.nodes_by_id.values():
            if node.last_activated_at == 0:
                idle_steps = max(0, current_timestamp - node.created_at)
            else:
                idle_steps = max(0, current_timestamp - node.last_activated_at)
            if idle_steps <= 0:
                continue
            connection_cost = (
                self.state.graph.degree_in(node.cell_id) + self.state.graph.degree_out(node.cell_id)
            ) * self.connection_cost_rate
            node.recent_activity = max(0.0, node.recent_activity - (0.08 * idle_steps))
            node.energy = max(0.0, node.energy - (((0.08 / 2.0) + connection_cost) * idle_steps))
            if node.energy <= self.dormancy_energy_threshold and idle_steps >= self.dormancy_idle_threshold:
                node.dormant = True

    def snapshot(self) -> RisaKernelSnapshot:
        return RisaKernelSnapshot(state=self.state.to_dict())

    def _build_graph_links(self, event: RisaObservation) -> None:
        actor = _normalize_label(event.actor)
        action = _normalize_label(event.action)
        event_id = f"event:{_normalize_label(event.event_id)}"
        actor_id = f"entity:{actor}"
        action_id = f"process:{action}"
        event_energy = max(event.event_energy, event.resonance_score, event.credit_longevity)

        self.state.graph.add_or_update_node(
            ConceptCell(
                cell_id=actor_id,
                kind="entity",
                label=actor,
                created_at=event.timestamp,
                recent_activity=1.0 + event_energy,
                energy=min(1.0, 0.35 + event_energy),
                last_activated_at=event.timestamp,
            )
        )
        self.state.graph.add_or_update_node(
            ConceptCell(
                cell_id=action_id,
                kind="process",
                label=action,
                created_at=event.timestamp,
                recent_activity=1.0 + event_energy,
                energy=min(1.0, 0.35 + event_energy),
                last_activated_at=event.timestamp,
            )
        )
        self.state.graph.add_or_update_node(
            ConceptCell(
                cell_id=event_id,
                kind="event",
                label=event.event_id,
                attributes={"source_ref": event.source_ref, "verified": str(event.verified).lower()},
                created_at=event.timestamp,
                recent_activity=1.0,
                energy=min(1.0, 0.3 + event_energy),
                last_activated_at=event.timestamp,
            )
        )
        self.state.graph.add_or_update_edge(
            ConceptRelation(
                source=actor_id,
                target=event_id,
                relation_type="participates_in",
                evidence_count=1,
                reliability=max(event.resonance_score, event.credit_longevity),
                last_updated=event.timestamp,
            )
        )
        self.state.graph.add_or_update_edge(
            ConceptRelation(
                source=event_id,
                target=action_id,
                relation_type="executes",
                evidence_count=1,
                reliability=max(event.resonance_score, event.credit_longevity),
                context_tags=tuple(sorted(_normalize_label(tag) for tag in event.context_tags)),
                last_updated=event.timestamp,
            )
        )
        for effect in event.observed_effects:
            effect_label = _normalize_label(effect)
            effect_id = f"state:{effect_label}"
            self.state.graph.add_or_update_node(
                ConceptCell(
                    cell_id=effect_id,
                    kind="state",
                    label=effect_label,
                    created_at=event.timestamp,
                    recent_activity=1.0 + event_energy,
                    energy=min(1.0, 0.35 + event_energy),
                    last_activated_at=event.timestamp,
                )
            )
            self.state.graph.add_or_update_edge(
                ConceptRelation(
                    source=action_id,
                    target=effect_id,
                    relation_type="predicts",
                    evidence_count=1 if event.verified else 0,
                    reliability=max(event.resonance_score, event.credit_longevity),
                    context_tags=tuple(sorted(_normalize_label(tag) for tag in event.context_tags)),
                    last_updated=event.timestamp,
                )
            )
            self.state.graph.add_or_update_edge(
                ConceptRelation(
                    source=event_id,
                    target=effect_id,
                    relation_type="observes",
                    evidence_count=1,
                    reliability=max(event.resonance_score, event.credit_longevity),
                    last_updated=event.timestamp,
                )
            )
        previous_id = self.state.previous_observation_id
        if previous_id and previous_id in self.state.observations_by_id:
            previous_action = _normalize_label(self.state.observations_by_id[previous_id].action)
            self.state.graph.add_or_update_edge(
                ConceptRelation(
                    source=f"process:{previous_action}",
                    target=action_id,
                    relation_type="precedes",
                    evidence_count=1,
                    reliability=max(event.resonance_score, event.credit_longevity),
                    context_tags=tuple(sorted(_normalize_label(tag) for tag in event.context_tags)),
                    last_updated=event.timestamp,
                )
            )

    def _learn_verified_pattern(self, event: RisaObservation) -> None:
        actor = _normalize_label(event.actor)
        action = _normalize_label(event.action)
        context_key = _context_key(event.context_tags)
        actor_bucket = self.state.actor_action_effect_counts.setdefault(actor, {})
        effect_bucket = actor_bucket.setdefault(action, {})
        action_bucket = self.state.action_effect_counts.setdefault(action, {})
        actor_context_bucket = self.state.actor_action_context_effect_counts.setdefault(actor, {}).setdefault(action, {})
        context_effect_bucket = actor_context_bucket.setdefault(context_key, {})
        action_context_bucket = self.state.action_context_effect_counts.setdefault(action, {})
        action_context_effect_bucket = action_context_bucket.setdefault(context_key, {})

        for effect in event.observed_effects:
            effect_label = _normalize_label(effect)
            effect_bucket[effect_label] = effect_bucket.get(effect_label, 0) + 1
            action_bucket[effect_label] = action_bucket.get(effect_label, 0) + 1
            context_effect_bucket[effect_label] = context_effect_bucket.get(effect_label, 0) + 1
            action_context_effect_bucket[effect_label] = action_context_effect_bucket.get(effect_label, 0) + 1

            pattern_id = f"pattern:{action}->{effect_label}"
            pattern = self.state.patterns.get(pattern_id)
            if pattern is None:
                pattern = ConceptPattern(pattern_id=pattern_id, signature=f"{action}->{effect_label}")
                self.state.patterns[pattern_id] = pattern
            pattern.event_count += 1
            pattern.support += 1
            pattern.verified_support += 1
            pattern.actors.add(actor)
            pattern.actions.add(action)
            pattern.effects.add(effect_label)
            pattern.context_tags.update(_normalize_label(tag) for tag in event.context_tags)

            _index_append(self.state.activation_index, f"actor:{actor}", effect_label)
            _index_append(self.state.activation_index, f"action:{action}", effect_label)
            _index_append(self.state.activation_index, f"context:{context_key}", effect_label)
            _index_append(self.state.activation_index, f"actor_action:{actor}:{action}", effect_label)

    def _rebuild_concepts(self, timestamp: int) -> None:
        for pattern in self.state.patterns.values():
            if pattern.support < self.min_support:
                continue
            if len(pattern.actors) < self.min_distinct_actors:
                continue
            action = next(iter(pattern.actions))
            effect = next(iter(pattern.effects))
            concept_id = f"concept:shared_{action}_{effect}"
            concept_label = f"shared_{action}_{effect}"
            support_energy = min(1.0, 0.4 + (0.1 * min(pattern.support, 4)))
            self.state.graph.add_or_update_node(
                ConceptCell(
                    cell_id=concept_id,
                    kind="concept",
                    label=concept_label,
                    attributes={"shared_action": action, "shared_effect": effect},
                    abstraction_level=1,
                    created_at=timestamp,
                    stability=min(1.0, float(pattern.support) / 5.0),
                    recent_activity=min(5.0, float(pattern.support)),
                    energy=support_energy,
                    last_activated_at=timestamp,
                )
            )
            self.state.concept_members[concept_id] = sorted(pattern.actors)
            self.state.concept_lineage[concept_id] = {
                "pattern_id": pattern.pattern_id,
                "support": pattern.support,
                "verified_support": pattern.verified_support,
                "actors": sorted(pattern.actors),
                "effects": sorted(pattern.effects),
            }
            _index_append(self.state.activation_index, f"concept:{concept_id}", effect)
            self.state.graph.add_or_update_edge(
                ConceptRelation(
                    source=concept_id,
                    target=f"process:{action}",
                    relation_type="participates_in",
                    evidence_count=pattern.support,
                    reliability=min(1.0, float(pattern.verified_support) / max(1, pattern.support)),
                    last_updated=timestamp,
                )
            )
            self.state.graph.add_or_update_edge(
                ConceptRelation(
                    source=concept_id,
                    target=f"state:{effect}",
                    relation_type="predicts",
                    evidence_count=pattern.support,
                    reliability=min(1.0, float(pattern.verified_support) / max(1, pattern.support)),
                    last_updated=timestamp,
                )
            )
            for actor in sorted(pattern.actors):
                self.state.graph.add_or_update_edge(
                    ConceptRelation(
                        source=f"entity:{actor}",
                        target=concept_id,
                        relation_type="instance_of",
                        evidence_count=pattern.support,
                        reliability=min(1.0, float(pattern.verified_support) / max(1, pattern.support)),
                        last_updated=timestamp,
                    )
                )
            concept = self.state.graph.get_node(concept_id)
            if concept is not None and self._should_prune_or_sleep(concept_id):
                concept.dormant = True

    def _should_prune_or_sleep(self, concept_id: str, *, min_energy: float = 0.08, max_connection_budget: int = 8) -> bool:
        node = self.state.graph.get_node(concept_id)
        if node is None:
            return False
        total_degree = self.state.graph.degree_in(concept_id) + self.state.graph.degree_out(concept_id)
        return node.energy <= min_energy and total_degree >= max_connection_budget

    def _collect_candidates(self, actor: str, action: str, context_key: str) -> List[str]:
        keys = [
            f"actor_action:{actor}:{action}",
            f"actor:{actor}",
            f"action:{action}",
            f"context:{context_key}",
        ]
        values = set()
        for key in keys:
            values.update(self.state.activation_index.get(key, []))
        candidates: List[str] = []
        for effect in sorted(values):
            node = self.state.graph.get_node(f"state:{effect}")
            if node is None or not node.dormant:
                candidates.append(effect)
        return candidates


def observation_from_record(record: Mapping[str, object], *, timestamp: int) -> RisaObservation:
    return RisaObservation(
        timestamp=int(timestamp),
        actor=str(record.get("actor", "")),
        action=str(record.get("action", "")),
        observed_effects=[str(item) for item in record.get("observed_effects", [])],
        event_id=str(record.get("event_id", "")),
        target=str(record.get("target", "")) if record.get("target") is not None else None,
        context_tags=[str(item) for item in record.get("context_tags", [])],
        source_ref=str(record.get("source_ref", "")),
        verified=bool(record.get("verified", True)),
        resonance_score=float(record.get("resonance_score", 0.0) or 0.0),
        credit_longevity=float(record.get("credit_longevity", 0.0) or 0.0),
        event_energy=float(record.get("event_energy", 0.0) or 0.0),
    )
