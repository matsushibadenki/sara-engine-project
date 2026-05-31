# Directory Path: src/sara_engine/nn/plastic_submodel_registry.py
# English Title: Plastic Submodel Registry
# Purpose/Content: Tracks bounded dynamic routing and local relearning traces for many specialized plastic submodels.

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


def _stable_id(text: str, modulus: int = 4096) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % max(1, int(modulus))


def _as_event_ids(events: Iterable[Any]) -> Set[int]:
    ids: Set[int] = set()
    for event in events:
        if hasattr(event, "spike_id"):
            ids.add(int(event.spike_id))
        elif isinstance(event, Mapping) and "spike_id" in event:
            ids.add(int(event["spike_id"]))
        else:
            ids.add(int(event))
    return ids


@dataclass
class PlasticSubmodel:
    """A compact specialist node with sparse concepts and local plasticity."""

    submodel_id: str
    role: str
    concepts: Tuple[str, ...]
    event_budget: int
    plasticity: float = 0.1
    active: bool = True
    concept_ids: Set[int] = field(default_factory=set)
    local_relearn_count: int = 0

    def __post_init__(self) -> None:
        self.event_budget = max(1, int(self.event_budget))
        self.plasticity = max(0.0, min(1.0, float(self.plasticity)))
        if not self.concept_ids:
            self.concept_ids = {_stable_id(f"{self.submodel_id}:{concept}") for concept in self.concepts}


class PlasticSubmodelRegistry:
    """Routes sparse events across small specialist submodels without dense global state."""

    def __init__(self, *, max_submodels: int = 16, max_route_edges: int = 32) -> None:
        self.max_submodels = max(1, int(max_submodels))
        self.max_route_edges = max(1, int(max_route_edges))
        self._submodels: Dict[str, PlasticSubmodel] = {}
        self._edges: Set[Tuple[str, str]] = set()
        self._trace: List[Dict[str, Any]] = []
        self._clock = 0

    def register(
        self,
        submodel_id: str,
        *,
        role: str,
        concepts: Sequence[str],
        event_budget: int,
        plasticity: float = 0.1,
    ) -> Dict[str, Any]:
        if len(self._submodels) >= self.max_submodels and submodel_id not in self._submodels:
            raise ValueError("submodel budget exceeded")
        submodel = PlasticSubmodel(
            submodel_id=str(submodel_id),
            role=str(role),
            concepts=tuple(str(concept) for concept in concepts),
            event_budget=int(event_budget),
            plasticity=float(plasticity),
        )
        self._submodels[submodel.submodel_id] = submodel
        return self._record("register", submodel.submodel_id, role=submodel.role)

    def connect(self, source_id: str, target_id: str, *, reason: str) -> Dict[str, Any]:
        self._require_submodel(source_id)
        self._require_submodel(target_id)
        if len(self._edges) >= self.max_route_edges and (source_id, target_id) not in self._edges:
            raise ValueError("route edge budget exceeded")
        self._edges.add((str(source_id), str(target_id)))
        return self._record("connect", f"{source_id}->{target_id}", reason=str(reason))

    def disconnect(self, source_id: str, target_id: str, *, reason: str) -> Dict[str, Any]:
        self._edges.discard((str(source_id), str(target_id)))
        return self._record("disconnect", f"{source_id}->{target_id}", reason=str(reason))

    def set_active(self, submodel_id: str, *, active: bool, reason: str) -> Dict[str, Any]:
        submodel = self._require_submodel(submodel_id)
        submodel.active = bool(active)
        event_type = "activate" if submodel.active else "deactivate"
        return self._record(event_type, submodel.submodel_id, reason=str(reason))

    def relearn_local(
        self,
        submodel_id: str,
        *,
        positive_events: Iterable[Any] = (),
        negative_events: Iterable[Any] = (),
        credit: float = 1.0,
    ) -> Dict[str, Any]:
        submodel = self._require_submodel(submodel_id)
        positive_ids = _as_event_ids(positive_events)
        negative_ids = _as_event_ids(negative_events)
        local_credit = max(-1.0, min(1.0, float(credit)))
        if local_credit >= 0.0:
            submodel.concept_ids = set(sorted(submodel.concept_ids | positive_ids)[: submodel.event_budget])
        else:
            submodel.concept_ids = set(submodel.concept_ids.difference(negative_ids))
        submodel.local_relearn_count += 1
        return self._record(
            "relearn",
            submodel.submodel_id,
            credit=local_credit,
            positive_event_count=len(positive_ids),
            negative_event_count=len(negative_ids),
            local_relearn_count=submodel.local_relearn_count,
        )

    def route(self, events: Iterable[Any], *, goal: str) -> Dict[str, Any]:
        event_ids = _as_event_ids(events)
        selected = []
        for submodel in sorted(self._submodels.values(), key=lambda item: item.submodel_id):
            if not submodel.active:
                continue
            overlap = len(event_ids.intersection(submodel.concept_ids))
            role_match = 1 if submodel.role in str(goal) or submodel.submodel_id in str(goal) else 0
            if overlap > 0 or role_match > 0:
                selected.append(
                    {
                        "submodel_id": submodel.submodel_id,
                        "role": submodel.role,
                        "overlap": overlap,
                        "role_match": role_match,
                        "event_budget": submodel.event_budget,
                        "concept_count": len(submodel.concept_ids),
                    }
                )
        connected_pairs = [
            {"source": source, "target": target}
            for source, target in sorted(self._edges)
            if any(item["submodel_id"] == source for item in selected)
            and any(item["submodel_id"] == target for item in selected)
        ]
        trace = self._record(
            "route",
            str(goal),
            selected_count=len(selected),
            connected_pair_count=len(connected_pairs),
        )
        return {
            "goal": str(goal),
            "event_count": len(event_ids),
            "selected_submodels": selected,
            "connected_pairs": connected_pairs,
            "trace_event": trace,
            "state_budget_ok": self.state_budget_ok(),
        }

    def apply_route_credit(
        self,
        route: Mapping[str, Any],
        *,
        support_events: Iterable[Any] = (),
        credit: float,
        reason: str,
    ) -> Dict[str, Any]:
        local_credit = max(-1.0, min(1.0, float(credit)))
        event_ids = _as_event_ids(support_events)
        selected_submodels = route.get("selected_submodels", [])
        if not isinstance(selected_submodels, list):
            selected_submodels = []
        updates = []
        for item in selected_submodels:
            if not isinstance(item, Mapping):
                continue
            submodel_id = str(item.get("submodel_id", "") or "")
            if not submodel_id or submodel_id not in self._submodels:
                continue
            fallback_id = _stable_id(f"route-credit:{submodel_id}:{reason}")
            update_events = sorted(event_ids) or [fallback_id]
            if local_credit >= 0.0:
                update = self.relearn_local(
                    submodel_id,
                    positive_events=update_events,
                    credit=local_credit,
                )
            else:
                update = self.relearn_local(
                    submodel_id,
                    negative_events=update_events,
                    credit=local_credit,
                )
            updates.append(update)
        trace = self._record(
            "route_credit",
            str(route.get("goal", "")),
            credit=local_credit,
            reason=str(reason),
            updated_submodel_count=len(updates),
        )
        return {
            "credit": local_credit,
            "reason": str(reason),
            "updated_submodel_count": len(updates),
            "updates": updates,
            "trace_event": trace,
            "state_budget_ok": self.state_budget_ok(),
        }

    def adapt_route_edges(
        self,
        route: Mapping[str, Any],
        *,
        credit: float,
        reason: str,
    ) -> Dict[str, Any]:
        local_credit = max(-1.0, min(1.0, float(credit)))
        selected_submodels = route.get("selected_submodels", [])
        if not isinstance(selected_submodels, list):
            selected_submodels = []
        selected_ids = [
            str(item.get("submodel_id", "") or "")
            for item in selected_submodels
            if isinstance(item, Mapping) and str(item.get("submodel_id", "") or "") in self._submodels
        ]
        connected_pairs = route.get("connected_pairs", [])
        if not isinstance(connected_pairs, list):
            connected_pairs = []

        created_edges: List[Dict[str, str]] = []
        pruned_edges: List[Dict[str, str]] = []
        if local_credit > 0.0 and len(selected_ids) >= 2:
            for source_id, target_id in zip(selected_ids, selected_ids[1:]):
                if len(self._edges) >= self.max_route_edges:
                    break
                if (source_id, target_id) in self._edges:
                    continue
                self._edges.add((source_id, target_id))
                created_edges.append({"source": source_id, "target": target_id})
        elif local_credit < 0.0:
            for pair in connected_pairs:
                if not isinstance(pair, Mapping):
                    continue
                source_id = str(pair.get("source", "") or "")
                target_id = str(pair.get("target", "") or "")
                if (source_id, target_id) not in self._edges:
                    continue
                self._edges.discard((source_id, target_id))
                pruned_edges.append({"source": source_id, "target": target_id})
                break

        trace = self._record(
            "route_edge_adaptation",
            str(route.get("goal", "")),
            credit=local_credit,
            reason=str(reason),
            created_edge_count=len(created_edges),
            pruned_edge_count=len(pruned_edges),
        )
        return {
            "credit": local_credit,
            "reason": str(reason),
            "created_edges": created_edges,
            "pruned_edges": pruned_edges,
            "trace_event": trace,
            "state_budget_ok": self.state_budget_ok(),
            "route_edge_count": len(self._edges),
            "route_edge_budget": self.max_route_edges,
        }

    def concept_trace(self) -> Dict[str, Any]:
        return {
            "schema": "sara-plastic-submodel-concept-trace-v1",
            "submodels": [
                {
                    "submodel_id": submodel.submodel_id,
                    "role": submodel.role,
                    "concept_count": len(submodel.concept_ids),
                    "local_relearn_count": submodel.local_relearn_count,
                    "active": submodel.active,
                }
                for submodel in sorted(self._submodels.values(), key=lambda item: item.submodel_id)
            ],
            "route_edges": [
                {"source": source, "target": target}
                for source, target in sorted(self._edges)
            ],
            "trace": list(self._trace),
        }

    def state_budget_ok(self) -> bool:
        return len(self._submodels) <= self.max_submodels and len(self._edges) <= self.max_route_edges

    def _require_submodel(self, submodel_id: str) -> PlasticSubmodel:
        submodel = self._submodels.get(str(submodel_id))
        if submodel is None:
            raise KeyError(f"unknown submodel: {submodel_id}")
        return submodel

    def _record(self, event_type: str, subject: str, **payload: Any) -> Dict[str, Any]:
        self._clock += 1
        event = {
            "timestep": self._clock,
            "event_type": str(event_type),
            "subject": str(subject),
            **payload,
        }
        self._trace.append(event)
        return event


def build_default_plastic_submodel_registry() -> PlasticSubmodelRegistry:
    """Builds a compact Stage E registry for runtime action grounding probes."""

    registry = PlasticSubmodelRegistry(max_submodels=8, max_route_edges=12)
    specialists = [
        ("world_model", "world", ("release", "causal_transition", "state"), 12),
        ("memory_system", "memory", ("pytest", "gate", "recall"), 12),
        ("value_system", "value", ("risk", "energy", "reward"), 8),
        ("body_control", "body", ("actuator", "sensor", "latency"), 8),
        ("language_system", "language", ("instruction", "summary", "dialogue"), 10),
        ("math_system", "math", ("budget", "score", "threshold"), 8),
        ("self_monitor", "self_monitor", ("audit", "trace", "uncertainty"), 10),
    ]
    for submodel_id, role, concepts, budget in specialists:
        registry.register(submodel_id, role=role, concepts=concepts, event_budget=budget, plasticity=0.2)

    registry.connect("memory_system", "world_model", reason="memory supports causal transition")
    registry.connect("world_model", "value_system", reason="world state informs value")
    registry.connect("value_system", "body_control", reason="value gates embodied action")
    registry.connect("language_system", "self_monitor", reason="language trace is audited")
    registry.disconnect("body_control", "language_system", reason="avoid unnecessary embodied feedback")
    return registry


def evaluate_plastic_submodel_registry_trace() -> Dict[str, Any]:
    registry = build_default_plastic_submodel_registry()
    registry.relearn_local(
        "world_model",
        positive_events=[_stable_id("release:ready"), _stable_id("pytest:passed")],
        credit=1.0,
    )
    registry.relearn_local(
        "self_monitor",
        positive_events=[_stable_id("audit:complete"), _stable_id("uncertainty:low")],
        credit=0.8,
    )
    registry.relearn_local(
        "value_system",
        negative_events=[_stable_id("risk:pytest_pending")],
        credit=-0.5,
    )
    route = registry.route(
        [_stable_id("release:ready"), _stable_id("audit:complete"), _stable_id("pytest:passed")],
        goal="world memory value self_monitor release",
    )
    concept_trace = registry.concept_trace()

    roles = {item["role"] for item in concept_trace["submodels"]}
    required_roles = {"world", "memory", "value", "body", "language", "math", "self_monitor"}
    event_types = {event["event_type"] for event in concept_trace["trace"]}
    selected_ids = {item["submodel_id"] for item in route["selected_submodels"]}
    metrics = {
        "plastic_submodel_registry_integrity": 1.0 if required_roles.issubset(roles) else 0.0,
        "dynamic_submodel_route_integrity": (
            1.0
            if {"connect", "disconnect", "route"}.issubset(event_types)
            and len(route["connected_pairs"]) >= 1
            else 0.0
        ),
        "submodel_relearning_trace_integrity": (
            1.0
            if sum(int(item["local_relearn_count"]) for item in concept_trace["submodels"]) >= 3
            and "relearn" in event_types
            else 0.0
        ),
        "interpretable_submodel_concept_trace": (
            1.0
            if bool(selected_ids)
            and all(int(item["concept_count"]) > 0 for item in concept_trace["submodels"])
            and bool(concept_trace["route_edges"])
            else 0.0
        ),
    }
    return {
        "observed_only": True,
        "metrics": metrics,
        "route": route,
        "concept_trace": concept_trace,
    }


def evaluate_plastic_submodel_intervention_trace() -> Dict[str, Any]:
    registry = build_default_plastic_submodel_registry()
    baseline = registry.route(
        [_stable_id("release:ready"), _stable_id("audit:complete"), _stable_id("pytest:passed")],
        goal="world memory value self_monitor release",
    )
    registry.set_active(
        "memory_system",
        active=False,
        reason="interpretability ablation: remove memory support",
    )
    ablated = registry.route(
        [_stable_id("release:ready"), _stable_id("audit:complete"), _stable_id("pytest:passed")],
        goal="world memory value self_monitor release",
    )
    registry.set_active(
        "memory_system",
        active=True,
        reason="interpretability restoration: recover memory support",
    )
    restored = registry.route(
        [_stable_id("release:ready"), _stable_id("audit:complete"), _stable_id("pytest:passed")],
        goal="world memory value self_monitor release",
    )
    concept_trace = registry.concept_trace()

    baseline_ids = {
        str(item.get("submodel_id", ""))
        for item in baseline.get("selected_submodels", [])
        if isinstance(item, Mapping)
    }
    ablated_ids = {
        str(item.get("submodel_id", ""))
        for item in ablated.get("selected_submodels", [])
        if isinstance(item, Mapping)
    }
    restored_ids = {
        str(item.get("submodel_id", ""))
        for item in restored.get("selected_submodels", [])
        if isinstance(item, Mapping)
    }
    event_types = {str(item.get("event_type", "")) for item in concept_trace.get("trace", [])}
    metrics = {
        "submodel_intervention_trace_integrity": (
            1.0 if {"deactivate", "activate", "route"}.issubset(event_types) else 0.0
        ),
        "submodel_ablation_effect_observed": (
            1.0
            if "memory_system" in baseline_ids
            and "memory_system" not in ablated_ids
            and len(ablated_ids) < len(baseline_ids)
            else 0.0
        ),
        "submodel_reactivation_recovery_observed": (
            1.0
            if "memory_system" in restored_ids
            and len(restored_ids) >= len(baseline_ids)
            else 0.0
        ),
    }
    return {
        "observed_only": True,
        "metrics": metrics,
        "baseline_route": baseline,
        "ablated_route": ablated,
        "restored_route": restored,
        "concept_trace": concept_trace,
    }


def evaluate_plastic_submodel_credit_assignment_trace() -> Dict[str, Any]:
    registry = build_default_plastic_submodel_registry()
    selected_route = registry.route(
        [_stable_id("release:ready"), _stable_id("pytest:passed")],
        goal="world memory value body math release primary",
    )
    counterfactual_route = registry.route(
        [_stable_id("release:blocked"), _stable_id("risk:pytest_pending")],
        goal="world memory value self_monitor language release counterfactual",
    )
    positive_feedback = registry.apply_route_credit(
        selected_route,
        support_events=[_stable_id("release:ready"), _stable_id("pytest:passed")],
        credit=0.9,
        reason="selected action succeeded",
    )
    negative_feedback = registry.apply_route_credit(
        counterfactual_route,
        support_events=[_stable_id("risk:pytest_pending")],
        credit=-0.6,
        reason="counterfactual branch rejected",
    )
    concept_trace = registry.concept_trace()
    route_credit_events = [
        item
        for item in concept_trace.get("trace", [])
        if isinstance(item, Mapping) and item.get("event_type") == "route_credit"
    ]
    selected_ids = {
        str(item.get("submodel_id", ""))
        for item in selected_route.get("selected_submodels", [])
        if isinstance(item, Mapping)
    }
    counterfactual_ids = {
        str(item.get("submodel_id", ""))
        for item in counterfactual_route.get("selected_submodels", [])
        if isinstance(item, Mapping)
    }
    metrics = {
        "submodel_credit_assignment_trace_integrity": (
            1.0
            if len(route_credit_events) == 2
            and positive_feedback["updated_submodel_count"] > 0
            and negative_feedback["updated_submodel_count"] > 0
            else 0.0
        ),
        "submodel_credit_selectivity_observed": (
            1.0 if selected_ids != counterfactual_ids and bool(selected_ids) and bool(counterfactual_ids) else 0.0
        ),
        "submodel_credit_state_budget_observed": (
            1.0
            if positive_feedback["state_budget_ok"]
            and negative_feedback["state_budget_ok"]
            and registry.state_budget_ok()
            else 0.0
        ),
    }
    return {
        "observed_only": True,
        "metrics": metrics,
        "selected_route": selected_route,
        "counterfactual_route": counterfactual_route,
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback,
        "concept_trace": concept_trace,
    }


def evaluate_plastic_submodel_structural_adaptation_trace() -> Dict[str, Any]:
    registry = build_default_plastic_submodel_registry()
    positive_route = registry.route(
        [_stable_id("release:ready"), _stable_id("pytest:passed")],
        goal="language self_monitor memory world value body math release",
    )
    growth = registry.adapt_route_edges(
        positive_route,
        credit=0.8,
        reason="successful route grows bounded support edge",
    )
    negative_route = registry.route(
        [_stable_id("risk:pytest_pending")],
        goal="memory world value release counterfactual",
    )
    pruning = registry.adapt_route_edges(
        negative_route,
        credit=-0.7,
        reason="failed route prunes one weak support edge",
    )
    concept_trace = registry.concept_trace()
    adaptation_events = [
        item
        for item in concept_trace.get("trace", [])
        if isinstance(item, Mapping) and item.get("event_type") == "route_edge_adaptation"
    ]
    metrics = {
        "submodel_structural_adaptation_trace_integrity": (
            1.0 if len(adaptation_events) == 2 else 0.0
        ),
        "submodel_structural_growth_bounded_observed": (
            1.0
            if growth["state_budget_ok"]
            and growth["route_edge_count"] <= growth["route_edge_budget"]
            and len(growth["created_edges"]) >= 1
            else 0.0
        ),
        "submodel_structural_pruning_observed": (
            1.0
            if pruning["state_budget_ok"]
            and pruning["route_edge_count"] <= pruning["route_edge_budget"]
            and len(pruning["pruned_edges"]) >= 1
            else 0.0
        ),
    }
    return {
        "observed_only": True,
        "metrics": metrics,
        "positive_route": positive_route,
        "negative_route": negative_route,
        "growth": growth,
        "pruning": pruning,
        "concept_trace": concept_trace,
    }


def evaluate_plastic_submodel_scientific_model_trace() -> Dict[str, Any]:
    registry = build_default_plastic_submodel_registry()
    support_route = registry.route(
        [_stable_id("release:ready"), _stable_id("pytest:passed")],
        goal="memory world value body math release primary",
    )
    counterexample_route = registry.route(
        [_stable_id("release:blocked"), _stable_id("risk:pytest_pending")],
        goal="memory world value self_monitor language release counterexample",
    )
    hypothesis = {
        "hypothesis_id": "stage-e-release-readiness-hypothesis",
        "cause": "pytest:passed",
        "effect": "release:ready",
        "support_submodels": [
            str(item.get("submodel_id", ""))
            for item in support_route.get("selected_submodels", [])
            if isinstance(item, Mapping) and str(item.get("submodel_id", ""))
        ],
        "confidence": 0.72,
        "status": "proposed",
    }
    prediction = {
        "hypothesis_id": hypothesis["hypothesis_id"],
        "predicted_effect": "release:ready",
        "route_goal": support_route.get("goal", ""),
        "trace_complete": bool(hypothesis["support_submodels"]),
    }
    falsification = {
        "hypothesis_id": hypothesis["hypothesis_id"],
        "observed_effect": "release:blocked",
        "counterexample_submodels": [
            str(item.get("submodel_id", ""))
            for item in counterexample_route.get("selected_submodels", [])
            if isinstance(item, Mapping) and str(item.get("submodel_id", ""))
        ],
        "falsified": True,
    }
    registry.apply_route_credit(
        support_route,
        support_events=[_stable_id("pytest:passed")],
        credit=0.4,
        reason="hypothesis received partial support",
    )
    revision_adaptation = registry.adapt_route_edges(
        counterexample_route,
        credit=-0.5,
        reason="counterexample weakens overconfident support edge",
    )
    revised_hypothesis = {
        **hypothesis,
        "confidence": 0.52,
        "status": "revised",
        "revision_reason": "counterexample observed: release blocked despite pytest context",
        "guard_condition": "risk:pytest_pending must be absent",
    }
    concept_trace = registry.concept_trace()
    event_types = {str(item.get("event_type", "")) for item in concept_trace.get("trace", [])}
    metrics = {
        "submodel_scientific_hypothesis_trace_integrity": (
            1.0
            if hypothesis["support_submodels"]
            and prediction["trace_complete"]
            and bool(revised_hypothesis["guard_condition"])
            else 0.0
        ),
        "submodel_counterexample_revision_observed": (
            1.0
            if falsification["falsified"]
            and revised_hypothesis["confidence"] < hypothesis["confidence"]
            and "route_edge_adaptation" in event_types
            else 0.0
        ),
        "submodel_scientific_model_budget_observed": (
            1.0
            if registry.state_budget_ok()
            and revision_adaptation["state_budget_ok"]
            and revision_adaptation["route_edge_count"] <= revision_adaptation["route_edge_budget"]
            else 0.0
        ),
    }
    return {
        "observed_only": True,
        "metrics": metrics,
        "support_route": support_route,
        "counterexample_route": counterexample_route,
        "hypothesis": hypothesis,
        "prediction": prediction,
        "falsification": falsification,
        "revised_hypothesis": revised_hypothesis,
        "revision_adaptation": revision_adaptation,
        "concept_trace": concept_trace,
    }


def evaluate_plastic_submodel_open_ended_hypothesis_bank_trace() -> Dict[str, Any]:
    registry = build_default_plastic_submodel_registry()
    route_specs = [
        {
            "hypothesis_id": "release-readiness",
            "events": [_stable_id("release:ready"), _stable_id("pytest:passed")],
            "goal": "memory world value body math release",
            "cause": "pytest:passed",
            "effect": "release:ready",
            "support": 0.86,
            "counterexample": False,
        },
        {
            "hypothesis_id": "risk-blocks-release",
            "events": [_stable_id("release:blocked"), _stable_id("risk:pytest_pending")],
            "goal": "memory world value self_monitor language release",
            "cause": "risk:pytest_pending",
            "effect": "release:blocked",
            "support": 0.80,
            "counterexample": False,
        },
        {
            "hypothesis_id": "language-only-release",
            "events": [_stable_id("instruction:release")],
            "goal": "language release",
            "cause": "instruction:release",
            "effect": "release:ready",
            "support": 0.38,
            "counterexample": True,
        },
        {
            "hypothesis_id": "audit-stabilizes-release",
            "events": [_stable_id("audit:complete"), _stable_id("release:ready")],
            "goal": "self_monitor memory world release",
            "cause": "audit:complete",
            "effect": "release:ready",
            "support": 0.74,
            "counterexample": False,
        },
    ]
    bank_capacity = 3
    candidates: List[Dict[str, Any]] = []
    for spec in route_specs:
        route = registry.route(spec["events"], goal=str(spec["goal"]))
        support_submodels = [
            str(item.get("submodel_id", ""))
            for item in route.get("selected_submodels", [])
            if isinstance(item, Mapping) and str(item.get("submodel_id", ""))
        ]
        novelty = len(set(support_submodels)) / max(len(support_submodels), 1)
        confidence = float(spec["support"]) - (0.35 if spec["counterexample"] else 0.0)
        candidates.append(
            {
                "hypothesis_id": str(spec["hypothesis_id"]),
                "cause": str(spec["cause"]),
                "effect": str(spec["effect"]),
                "support_submodels": support_submodels,
                "confidence": max(0.0, min(1.0, confidence)),
                "novelty": float(novelty),
                "counterexample_seen": bool(spec["counterexample"]),
                "route": route,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            bool(item["counterexample_seen"]),
            -float(item["confidence"]),
            -float(item["novelty"]),
            str(item["hypothesis_id"]),
        ),
    )
    retained = ranked[:bank_capacity]
    pruned = ranked[bank_capacity:]
    for item in retained:
        registry.apply_route_credit(
            item["route"],
            support_events=[_stable_id(str(item["cause"])), _stable_id(str(item["effect"]))],
            credit=float(item["confidence"]),
            reason=f"hypothesis bank retained {item['hypothesis_id']}",
        )
    for item in pruned:
        registry.adapt_route_edges(
            item["route"],
            credit=-0.4,
            reason=f"hypothesis bank pruned {item['hypothesis_id']}",
        )

    concept_trace = registry.concept_trace()
    event_types = {str(item.get("event_type", "")) for item in concept_trace.get("trace", [])}
    retained_ids = {str(item["hypothesis_id"]) for item in retained}
    pruned_ids = {str(item["hypothesis_id"]) for item in pruned}
    metrics = {
        "submodel_hypothesis_bank_integrity": (
            1.0
            if len(retained) == bank_capacity
            and len(pruned) == max(0, len(candidates) - bank_capacity)
            and all(item["support_submodels"] for item in retained)
            else 0.0
        ),
        "submodel_open_ended_selection_observed": (
            1.0
            if "language-only-release" in pruned_ids
            and {"release-readiness", "risk-blocks-release"}.issubset(retained_ids)
            else 0.0
        ),
        "submodel_hypothesis_bank_budget_observed": (
            1.0
            if len(retained) <= bank_capacity
            and registry.state_budget_ok()
            and {"route_credit", "route_edge_adaptation"}.issubset(event_types)
            else 0.0
        ),
    }
    return {
        "observed_only": True,
        "metrics": metrics,
        "bank_capacity": bank_capacity,
        "candidates": candidates,
        "retained_hypotheses": retained,
        "pruned_hypotheses": pruned,
        "concept_trace": concept_trace,
    }
