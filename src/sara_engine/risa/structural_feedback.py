from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

from sara_engine.learning.structural_plasticity import (
    BoundedStructuralPlasticityController,
    RouteKey,
    StructuralPlasticityResult,
)
from sara_engine.memory.concept_review_loop import ConceptReviewLoopResult

from .feedback import RisaFeedbackPackage
from .kernel import SARAAlignedRisaKernel

RELATION_TYPE_PROFILES: Dict[str, Dict[str, float]] = {
    "predicts": {
        "seed_responsibility_floor": 0.50,
        "seed_longevity_floor": 0.45,
        "support_gain_scale": 1.00,
        "replay_scale": 1.00,
        "growth_coactivation_scale": 1.00,
        "growth_prediction_scale": 1.00,
        "dormant_contradiction": 1.00,
    },
    "instance_of": {
        "seed_responsibility_floor": 0.35,
        "seed_longevity_floor": 0.60,
        "support_gain_scale": 0.65,
        "replay_scale": 0.85,
        "growth_coactivation_scale": 0.45,
        "growth_prediction_scale": 0.55,
        "dormant_contradiction": 0.35,
    },
    "precedes": {
        "seed_responsibility_floor": 0.25,
        "seed_longevity_floor": 0.30,
        "support_gain_scale": 0.75,
        "replay_scale": 0.55,
        "growth_coactivation_scale": 0.55,
        "growth_prediction_scale": 0.80,
        "dormant_contradiction": 0.50,
    },
    "participates_in": {
        "seed_responsibility_floor": 0.30,
        "seed_longevity_floor": 0.45,
        "support_gain_scale": 0.60,
        "replay_scale": 0.70,
        "growth_coactivation_scale": 0.30,
        "growth_prediction_scale": 0.40,
        "dormant_contradiction": 0.25,
    },
    "executes": {
        "seed_responsibility_floor": 0.30,
        "seed_longevity_floor": 0.40,
        "support_gain_scale": 0.60,
        "replay_scale": 0.65,
        "growth_coactivation_scale": 0.35,
        "growth_prediction_scale": 0.45,
        "dormant_contradiction": 0.25,
    },
    "observes": {
        "seed_responsibility_floor": 0.20,
        "seed_longevity_floor": 0.20,
        "support_gain_scale": 0.35,
        "replay_scale": 0.35,
        "growth_coactivation_scale": 0.20,
        "growth_prediction_scale": 0.25,
        "dormant_contradiction": 0.20,
    },
}


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _route_index(label: str, *, modulus: int = 100_003) -> int:
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % max(2, int(modulus))


def route_key_for_edge(source_id: str, target_id: str) -> RouteKey:
    return (_route_index(source_id), _route_index(target_id))


def route_key_for_relation(source_id: str, target_id: str, relation_type: str) -> RouteKey:
    return (
        _route_index(f"{relation_type}::{source_id}"),
        _route_index(f"{relation_type}::{target_id}"),
    )


def _relation_profile(relation_type: str) -> Dict[str, float]:
    return RELATION_TYPE_PROFILES.get(str(relation_type), RELATION_TYPE_PROFILES["predicts"])


def _relation_type_from_concept_key(value: str) -> str:
    text = str(value or "")
    if ":" not in text or "->" not in text:
        return ""
    return text.split(":", 1)[0].strip() or ""


def _phase_rank(phase: str) -> float:
    return {
        "liquid": 0.0,
        "glass": 0.5,
        "crystal": 1.0,
    }.get(str(phase or ""), 0.0)


def _default_relation_class_feedback() -> Dict[str, float]:
    return {
        "ready_count": 0.0,
        "blocked_count": 0.0,
        "review_support_mean": 0.0,
        "contradiction_mean": 0.0,
        "historical_contradiction_mean": 0.0,
        "historical_contradiction_peak": 0.0,
        "contradiction_persistence": 0.0,
        "replay_score_mean": 0.0,
        "energy_cost_mean": 0.0,
        "energy_pressure": 0.0,
        "phase_maturity_mean": 0.0,
        "phase_plasticity_mean": 1.0,
        "phase_retention_mean": 0.0,
        "stability_support_multiplier": 1.0,
        "growth_support_multiplier": 1.0,
        "active_route_multiplier": 1.0,
        "prune_pressure": 0.0,
    }


def _build_relation_class_feedback(
    review_result: ConceptReviewLoopResult,
    *,
    controller: BoundedStructuralPlasticityController,
    route_labels: Mapping[str, Mapping[str, str]],
    idle_replay_report: Mapping[str, Any] | None = None,
    memory_phase_report: Mapping[str, Any] | None = None,
) -> Dict[str, Dict[str, float]]:
    aggregates: Dict[str, Dict[str, float]] = {}

    def bucket(relation_type: str) -> Dict[str, float]:
        relation_key = str(relation_type or "").strip() or "predicts"
        return aggregates.setdefault(
            relation_key,
            {
                "ready_count": 0.0,
                "blocked_count": 0.0,
                "review_support_sum": 0.0,
                "review_support_seen": 0.0,
                "contradiction_sum": 0.0,
                "contradiction_seen": 0.0,
                "historical_contradiction_sum": 0.0,
                "historical_contradiction_seen": 0.0,
                "historical_contradiction_peak": 0.0,
                "replay_score_sum": 0.0,
                "replay_seen": 0.0,
                "energy_cost_sum": 0.0,
                "energy_seen": 0.0,
                "phase_maturity_sum": 0.0,
                "phase_plasticity_sum": 0.0,
                "phase_retention_sum": 0.0,
                "phase_seen": 0.0,
            },
        )

    for route_key, route_state in controller.routes.items():
        label = route_labels.get(f"{route_key[0]}:{route_key[1]}", {})
        relation_type = str(label.get("relation_type", "") or "")
        if not relation_type:
            continue
        entry = bucket(relation_type)
        historical_contradiction = min(
            1.0,
            float(route_state.contradiction_count)
            / float(max(1, controller.contradiction_prune_threshold)),
        )
        entry["historical_contradiction_sum"] += historical_contradiction
        entry["historical_contradiction_seen"] += 1.0
        entry["historical_contradiction_peak"] = max(
            entry["historical_contradiction_peak"],
            historical_contradiction,
        )

    for decision in review_result.schedule.ready_queue:
        relation_type = _relation_type_from_concept_key(str(decision.concept_key))
        if not relation_type:
            continue
        entry = bucket(relation_type)
        entry["ready_count"] += 1.0
        entry["review_support_sum"] += max(
            0.0,
            min(
                1.0,
                0.45 * float(decision.priority_score)
                + 0.25 * float(decision.credit_score)
                + 0.15 * float(decision.credit_longevity)
                + 0.10 * float(decision.self_state_alignment_score)
                + 0.05 * float(decision.multimodal_bundle_affinity),
            ),
        )
        entry["review_support_seen"] += 1.0
        entry["contradiction_sum"] += _clamp01(float(decision.contradiction_score))
        entry["contradiction_seen"] += 1.0

    for decision in review_result.schedule.blocked_queue:
        relation_type = _relation_type_from_concept_key(str(decision.concept_key))
        if not relation_type:
            continue
        entry = bucket(relation_type)
        entry["blocked_count"] += 1.0
        entry["contradiction_sum"] += _clamp01(float(decision.contradiction_score))
        entry["contradiction_seen"] += 1.0

    selected_candidates = tuple((idle_replay_report or {}).get("selected", ()))
    selected_by_memory_id: Dict[str, Mapping[str, Any]] = {}
    for candidate in selected_candidates:
        relation_type = _relation_type_from_concept_key(str(candidate.get("own_latent_id", "")))
        if not relation_type:
            continue
        entry = bucket(relation_type)
        replay_score = _clamp01(float(candidate.get("replay_score", 0.0) or 0.0))
        energy_cost = max(0.0, float(candidate.get("event_cost", 0.0) or 0.0))
        entry["replay_score_sum"] += replay_score
        entry["replay_seen"] += 1.0
        entry["energy_cost_sum"] += energy_cost
        entry["energy_seen"] += 1.0
        memory_id = str(candidate.get("entry_id", "") or "")
        if memory_id:
            selected_by_memory_id[memory_id] = candidate

    for track in tuple((memory_phase_report or {}).get("phase_tracks", ())):
        candidate = selected_by_memory_id.get(str(track.get("memory_id", "") or ""))
        if candidate is None:
            continue
        relation_type = _relation_type_from_concept_key(str(candidate.get("own_latent_id", "")))
        if not relation_type:
            continue
        entry = bucket(relation_type)
        entry["phase_maturity_sum"] += _phase_rank(str(track.get("final_phase", "")))
        entry["phase_plasticity_sum"] += _clamp01(float(track.get("final_plasticity", 1.0) or 0.0))
        entry["phase_retention_sum"] += _clamp01(float(track.get("final_retention", 0.0) or 0.0))
        entry["phase_seen"] += 1.0

    feedback: Dict[str, Dict[str, float]] = {}
    for relation_type, entry in aggregates.items():
        review_support_mean = (
            entry["review_support_sum"] / entry["review_support_seen"]
            if entry["review_support_seen"] > 0.0
            else 0.0
        )
        contradiction_mean = (
            entry["contradiction_sum"] / entry["contradiction_seen"]
            if entry["contradiction_seen"] > 0.0
            else 0.0
        )
        historical_contradiction_mean = (
            entry["historical_contradiction_sum"] / entry["historical_contradiction_seen"]
            if entry["historical_contradiction_seen"] > 0.0
            else 0.0
        )
        historical_contradiction_peak = entry["historical_contradiction_peak"]
        contradiction_persistence = _clamp01(
            max(
                0.55 * historical_contradiction_mean + 0.45 * contradiction_mean,
                0.55 * historical_contradiction_peak + 0.45 * contradiction_mean,
            )
        )
        replay_score_mean = (
            entry["replay_score_sum"] / entry["replay_seen"]
            if entry["replay_seen"] > 0.0
            else 0.0
        )
        energy_cost_mean = (
            entry["energy_cost_sum"] / entry["energy_seen"]
            if entry["energy_seen"] > 0.0
            else 0.0
        )
        phase_maturity_mean = (
            entry["phase_maturity_sum"] / entry["phase_seen"]
            if entry["phase_seen"] > 0.0
            else 0.0
        )
        phase_plasticity_mean = (
            entry["phase_plasticity_sum"] / entry["phase_seen"]
            if entry["phase_seen"] > 0.0
            else 1.0
        )
        phase_retention_mean = (
            entry["phase_retention_sum"] / entry["phase_seen"]
            if entry["phase_seen"] > 0.0
            else 0.0
        )
        energy_pressure = _clamp01(energy_cost_mean / 12.0)
        stability_support_multiplier = max(
            0.45,
            min(
                1.35,
                0.80
                + 0.30 * review_support_mean
                + 0.22 * phase_maturity_mean
                + 0.18 * phase_retention_mean
                + 0.10 * replay_score_mean
                - 0.35 * contradiction_persistence
                - 0.12 * energy_pressure,
            ),
        )
        growth_support_multiplier = max(
            0.35,
            min(
                1.30,
                0.72
                + 0.28 * replay_score_mean
                + 0.24 * phase_plasticity_mean
                + 0.16 * review_support_mean
                - 0.32 * contradiction_persistence
                - 0.18 * energy_pressure,
            ),
        )
        active_route_multiplier = max(
            0.15,
            min(
                1.20,
                0.70
                + 0.20 * replay_score_mean
                + 0.18 * phase_retention_mean
                + 0.10 * review_support_mean
                - 0.30 * contradiction_persistence
                - 0.15 * energy_pressure,
            ),
        )
        prune_pressure = max(
            0.0,
            min(
                0.95,
                0.45 * contradiction_persistence
                + 0.22 * energy_pressure
                + 0.15 * (1.0 - phase_retention_mean)
                - 0.10 * replay_score_mean
                - 0.06 * review_support_mean,
            ),
        )
        feedback[relation_type] = {
            "ready_count": float(entry["ready_count"]),
            "blocked_count": float(entry["blocked_count"]),
            "review_support_mean": round(review_support_mean, 6),
            "contradiction_mean": round(contradiction_mean, 6),
            "historical_contradiction_mean": round(historical_contradiction_mean, 6),
            "historical_contradiction_peak": round(historical_contradiction_peak, 6),
            "contradiction_persistence": round(contradiction_persistence, 6),
            "replay_score_mean": round(replay_score_mean, 6),
            "energy_cost_mean": round(energy_cost_mean, 6),
            "energy_pressure": round(energy_pressure, 6),
            "phase_maturity_mean": round(phase_maturity_mean, 6),
            "phase_plasticity_mean": round(phase_plasticity_mean, 6),
            "phase_retention_mean": round(phase_retention_mean, 6),
            "stability_support_multiplier": round(stability_support_multiplier, 6),
            "growth_support_multiplier": round(growth_support_multiplier, 6),
            "active_route_multiplier": round(active_route_multiplier, 6),
            "prune_pressure": round(prune_pressure, 6),
        }
    return feedback


@dataclass(frozen=True)
class RisaStructuralPlasticityCycleResult:
    structural_result: StructuralPlasticityResult
    route_labels: Dict[str, Dict[str, str]]
    signals: Dict[str, Any]
    support_route_count: int
    candidate_route_count: int
    schema: str = "sara-risa-structural-plasticity-cycle-result-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "structural_result": self.structural_result.to_dict(),
            "route_labels": dict(self.route_labels),
            "signals": dict(self.signals),
            "support_route_count": int(self.support_route_count),
            "candidate_route_count": int(self.candidate_route_count),
        }


def seed_structural_routes_from_risa(
    controller: BoundedStructuralPlasticityController,
    kernel: SARAAlignedRisaKernel,
) -> Dict[str, Dict[str, str]]:
    route_labels: Dict[str, Dict[str, str]] = {}
    for edge in kernel.state.graph.edges_by_key.values():
        profile = _relation_profile(edge.relation_type)
        route_key = route_key_for_relation(edge.source, edge.target, edge.relation_type)
        label_key = f"{route_key[0]}:{route_key[1]}"
        route_labels[label_key] = {
            "source": str(edge.source),
            "target": str(edge.target),
            "relation_type": str(edge.relation_type),
        }
        if route_key in controller.routes:
            continue
        source_node = kernel.state.graph.get_node(edge.source)
        target_node = kernel.state.graph.get_node(edge.target)
        dormant = bool(
            (source_node is not None and source_node.dormant)
            or (target_node is not None and target_node.dormant)
        )
        route_state = "decaying" if dormant else (
            "stable" if int(edge.evidence_count) >= controller.min_stable_verified_support else "provisional"
        )
        support_ratio = min(1.0, float(edge.evidence_count) / float(max(1, controller.min_stable_verified_support + 1)))
        energy = max(
            _clamp01(getattr(source_node, "energy", 0.0)),
            _clamp01(getattr(target_node, "energy", 0.0)),
        )
        controller.register_route(
            route_key,
            weight=float(edge.reliability),
            route_state=route_state,
            responsibility=max(_clamp01(edge.reliability), profile["seed_responsibility_floor"] * support_ratio),
            longevity=max(_clamp01(edge.reliability), profile["seed_longevity_floor"] * energy),
            prediction_gain_support=_clamp01(edge.reliability * profile["support_gain_scale"]),
            contradiction_count=int(profile["dormant_contradiction"]) if dormant else 0,
            support_count=max(1, int(edge.evidence_count)),
            verified_support_count=max(0, int(edge.evidence_count)),
            last_active_step=max(0, int(edge.last_updated)),
            created_step=max(0, int(edge.last_updated) - max(0, int(edge.evidence_count) - 1)),
        )
    return route_labels


def run_risa_structural_plasticity_cycle(
    controller: BoundedStructuralPlasticityController,
    kernel: SARAAlignedRisaKernel,
    *,
    review_result: ConceptReviewLoopResult,
    feedback_package: RisaFeedbackPackage | None = None,
    current_segment: int,
    idle_replay_report: Mapping[str, Any] | None = None,
    memory_phase_report: Mapping[str, Any] | None = None,
    frozen_evaluation: bool = False,
) -> RisaStructuralPlasticityCycleResult:
    route_labels = seed_structural_routes_from_risa(controller, kernel)
    active_routes: Dict[RouteKey, float] = {}
    event_memory_support: MutableMapping[RouteKey, Dict[str, Any]] = {}
    candidate_routes: MutableMapping[RouteKey, Dict[str, Any]] = {}
    relation_class_feedback = _build_relation_class_feedback(
        review_result,
        controller=controller,
        route_labels=route_labels,
        idle_replay_report=idle_replay_report,
        memory_phase_report=memory_phase_report,
    )

    for edge in kernel.state.graph.edges_by_key.values():
        profile = _relation_profile(edge.relation_type)
        class_feedback = relation_class_feedback.get(
            edge.relation_type,
            _default_relation_class_feedback(),
        )
        route_key = route_key_for_relation(edge.source, edge.target, edge.relation_type)
        source_node = kernel.state.graph.get_node(edge.source)
        target_node = kernel.state.graph.get_node(edge.target)
        dormant = bool(
            (source_node is not None and source_node.dormant)
            or (target_node is not None and target_node.dormant)
        )
        if dormant:
            continue
        freshness = _clamp01(1.0 - (max(0, int(current_segment) - int(edge.last_updated)) / 8.0))
        edge_activity = max(_clamp01(edge.reliability), freshness)
        edge_activity = _clamp01(
            edge_activity * float(class_feedback["active_route_multiplier"])
            - 0.20 * float(class_feedback["prune_pressure"])
        )
        active_routes[route_key] = max(
            active_routes.get(route_key, 0.0),
            edge_activity,
        )
        support = event_memory_support.setdefault(
            route_key,
            {
                "prediction_gain_support": 0.0,
                "replay_support": 0.0,
                "verified": False,
            },
        )
        support["prediction_gain_support"] = max(
            _clamp01(support["prediction_gain_support"]),
            _clamp01(
                edge.reliability
                * profile["support_gain_scale"]
                * float(class_feedback["stability_support_multiplier"])
            ),
        )
        support["replay_support"] = max(
            _clamp01(support["replay_support"]),
            _clamp01(
                profile["replay_scale"]
                * max(
                    _clamp01(getattr(source_node, "stability", 0.0)),
                    _clamp01(getattr(target_node, "stability", 0.0)),
                    _clamp01(getattr(source_node, "energy", 0.0)),
                    _clamp01(getattr(target_node, "energy", 0.0)),
                )
                * float(class_feedback["stability_support_multiplier"])
            ),
        )
        support["verified"] = bool(support["verified"]) or int(edge.evidence_count) > 0

    for decision in review_result.schedule.ready_queue:
        relation_type = _relation_type_from_concept_key(str(decision.concept_key))
        if not relation_type:
            continue
        relation_part = str(decision.concept_key)[len(f"{relation_type}:") :]
        if "->" not in relation_part:
            continue
        source_id, target_id = relation_part.split("->", 1)
        route_key = route_key_for_relation(source_id, target_id, relation_type)
        label_key = f"{route_key[0]}:{route_key[1]}"
        route_labels[label_key] = {
            "source": source_id,
            "target": target_id,
            "relation_type": relation_type,
        }
        class_feedback = relation_class_feedback.get(
            relation_type,
            _default_relation_class_feedback(),
        )
        support = event_memory_support.setdefault(
            route_key,
            {"prediction_gain_support": 0.0, "replay_support": 0.0, "verified": False},
        )
        support["prediction_gain_support"] = max(
            _clamp01(support["prediction_gain_support"]),
            _clamp01(
                (0.5 * decision.priority_score + 0.5 * decision.credit_score)
                * float(class_feedback["stability_support_multiplier"])
            ),
        )
        support["replay_support"] = max(
            _clamp01(support["replay_support"]),
            _clamp01(
                max(decision.credit_longevity, decision.self_state_alignment_score)
                * float(class_feedback["stability_support_multiplier"])
            ),
        )
        support["verified"] = True

    package = feedback_package
    if package is None:
        from .feedback import build_feedback_package

        package = build_feedback_package(kernel, current_segment=int(current_segment))
    for relation in package.candidate_relations:
        profile = _relation_profile(relation.relation)
        class_feedback = relation_class_feedback.get(
            relation.relation,
            _default_relation_class_feedback(),
        )
        route_key = route_key_for_relation(
            relation.source_event_id,
            relation.target_event_id,
            relation.relation,
        )
        label_key = f"{route_key[0]}:{route_key[1]}"
        route_labels[label_key] = {
            "source": str(relation.source_event_id),
            "target": str(relation.target_event_id),
            "relation_type": str(relation.relation),
        }
        candidate = candidate_routes.setdefault(
            route_key,
            {
                "coactivation": 0.0,
                "prediction_gain_support": 0.0,
                "verified": False,
                "weight": 0.0,
                "responsibility": 0.0,
                "longevity": 0.0,
            },
        )
        candidate["coactivation"] = max(
            _clamp01(candidate["coactivation"]),
            _clamp01(
                relation.confidence
                * profile["growth_coactivation_scale"]
                * float(class_feedback["growth_support_multiplier"])
            ),
        )
        candidate["prediction_gain_support"] = max(
            _clamp01(candidate["prediction_gain_support"]),
            _clamp01(
                relation.prediction_gain
                * profile["growth_prediction_scale"]
                * float(class_feedback["growth_support_multiplier"])
            ),
        )
        candidate["verified"] = True
        candidate["weight"] = max(float(candidate["weight"]), float(relation.confidence))
        candidate["responsibility"] = max(
            _clamp01(candidate["responsibility"]),
            _clamp01(
                relation.prediction_gain
                * profile["growth_prediction_scale"]
                * float(class_feedback["growth_support_multiplier"])
            ),
        )
        candidate["longevity"] = max(
            _clamp01(candidate["longevity"]),
            min(
                1.0,
                profile["seed_longevity_floor"]
                * float(relation.evidence_count)
                / 2.0
                * float(class_feedback["stability_support_multiplier"]),
            ),
        )
        candidate["contradiction_pressure"] = max(
            _clamp01(candidate.get("contradiction_pressure", 0.0)),
            _clamp01(class_feedback["contradiction_persistence"]),
        )

    selected = tuple((idle_replay_report or {}).get("selected", ()))
    admitted_count = len(review_result.admission_plan.admitted_candidates)
    ready_count = len(review_result.schedule.ready_queue)
    blocked_count = len(review_result.schedule.blocked_queue)
    blocked_contradiction = 0.0
    if blocked_count:
        blocked_contradiction = sum(
            float(item.contradiction_score) for item in review_result.schedule.blocked_queue
        ) / float(blocked_count)
    average_energy = 1.0
    if kernel.state.graph.nodes_by_id:
        average_energy = sum(
            _clamp01(node.energy) for node in kernel.state.graph.nodes_by_id.values() if not bool(node.dormant)
        ) / float(max(1, sum(1 for node in kernel.state.graph.nodes_by_id.values() if not bool(node.dormant))))
    signals = {
        "prediction_error": _clamp01(
            max(
                blocked_contradiction,
                0.8 if blocked_count > 0 and admitted_count == 0 else 0.0,
            )
        ),
        "novelty": _clamp01(
            max(
                float(len(package.candidate_relations)) / 4.0,
                float(len(selected)) / 4.0,
                0.4 if ready_count > 0 else 0.0,
            )
        ),
        "reward": _clamp01(float(admitted_count) / float(max(1, ready_count))),
        "contradiction": _clamp01(blocked_contradiction),
        "metabolic_headroom": _clamp01(average_energy),
        "source_backed": bool(package.candidate_relations or ready_count > 0),
        "abstained": False,
        "relation_class_feedback": relation_class_feedback,
    }
    structural_result = controller.apply_event(
        active_routes=active_routes,
        signals=signals,
        event_memory_support=event_memory_support,
        candidate_routes=candidate_routes,
        route_contradiction_pressure={
            route_key: _clamp01(
                relation_class_feedback.get(
                    str(label.get("relation_type", "")),
                    _default_relation_class_feedback(),
                )["contradiction_persistence"]
            )
            for route_key, label in (
                (
                    route_key,
                    route_labels.get(f"{route_key[0]}:{route_key[1]}", {}),
                )
                for route_key in controller.routes
            )
        },
        frozen_evaluation=bool(frozen_evaluation),
    )
    return RisaStructuralPlasticityCycleResult(
        structural_result=structural_result,
        route_labels=route_labels,
        signals=signals,
        support_route_count=len(event_memory_support),
        candidate_route_count=len(candidate_routes),
    )
