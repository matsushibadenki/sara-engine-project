from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


RouteKey = Tuple[int, int]
ROUTE_STATES = {"provisional", "stable", "decaying"}


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _route_weight(value: Any) -> float:
    if isinstance(value, tuple):
        return float(value[0] or 0.0)
    return float(value or 0.0)


@dataclass(frozen=True)
class StructuralRouteState:
    weight: float = 0.0
    route_state: str = "provisional"
    responsibility: float = 0.0
    longevity: float = 0.0
    prediction_gain_support: float = 0.0
    contradiction_count: int = 0
    support_count: int = 0
    verified_support_count: int = 0
    last_active_step: int = 0
    created_step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight": round(self.weight, 6),
            "route_state": self.route_state,
            "responsibility": round(self.responsibility, 6),
            "longevity": round(self.longevity, 6),
            "prediction_gain_support": round(self.prediction_gain_support, 6),
            "contradiction_count": self.contradiction_count,
            "support_count": self.support_count,
            "verified_support_count": self.verified_support_count,
            "last_active_step": self.last_active_step,
            "created_step": self.created_step,
        }


@dataclass(frozen=True)
class StructuralPlasticityResult:
    decision: str
    update_allowed: bool
    event_triggered: bool
    pruned_routes: Tuple[RouteKey, ...]
    grown_routes: Tuple[RouteKey, ...]
    stabilized_routes: Tuple[RouteKey, ...]
    decaying_routes: Tuple[RouteKey, ...]
    event_cost: int
    state_budget_units: int
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "update_allowed": self.update_allowed,
            "event_triggered": self.event_triggered,
            "pruned_routes": [f"{pre}:{post}" for pre, post in self.pruned_routes],
            "grown_routes": [f"{pre}:{post}" for pre, post in self.grown_routes],
            "stabilized_routes": [f"{pre}:{post}" for pre, post in self.stabilized_routes],
            "decaying_routes": [f"{pre}:{post}" for pre, post in self.decaying_routes],
            "event_cost": self.event_cost,
            "state_budget_units": self.state_budget_units,
            "trace": dict(self.trace),
        }


class StructuralPlasticityManager:
    """Legacy structural pruning helper kept for compatibility."""

    def __init__(self, prune_threshold: float = 0.1, growth_rate: float = 0.01) -> None:
        self.prune_threshold = float(prune_threshold)
        self.growth_rate = float(growth_rate)

    def should_prune(self, weight: float) -> bool:
        return abs(float(weight)) < self.prune_threshold

    def prune_synapses(
        self,
        synapses: list,
        max_size: int,
        target_size: int,
    ) -> None:
        for synapse_dict in synapses:
            if len(synapse_dict) > max_size:
                sorted_items = sorted(
                    synapse_dict.items(),
                    key=lambda item: abs(_route_weight(item[1])),
                )
                to_remove = len(synapse_dict) - int(target_size)
                for key, _ in sorted_items[:to_remove]:
                    del synapse_dict[key]


class BoundedStructuralPlasticityController:
    """Budgeted sparse rewiring controller for long-horizon local adaptation."""

    def __init__(
        self,
        *,
        max_total_links: int = 128,
        max_fan_in: int = 8,
        max_fan_out: int = 8,
        max_rewrites_per_event: int = 4,
        prune_grace_steps: int = 3,
        event_threshold: float = 0.45,
        metabolic_threshold: float = 0.2,
        contradiction_growth_block: float = 0.55,
        min_active_responsibility: float = 0.18,
        min_stable_prediction_gain: float = 0.65,
        min_stable_verified_support: int = 2,
        contradiction_prune_threshold: int = 2,
        provisional_growth_threshold: float = 0.7,
        responsibility_decay: float = 0.92,
        longevity_increment: float = 0.08,
        activity_increment: float = 0.3,
    ) -> None:
        self.max_total_links = max(1, int(max_total_links))
        self.max_fan_in = max(1, int(max_fan_in))
        self.max_fan_out = max(1, int(max_fan_out))
        self.max_rewrites_per_event = max(1, int(max_rewrites_per_event))
        self.prune_grace_steps = max(1, int(prune_grace_steps))
        self.event_threshold = _clamp01(event_threshold)
        self.metabolic_threshold = _clamp01(metabolic_threshold)
        self.contradiction_growth_block = _clamp01(contradiction_growth_block)
        self.min_active_responsibility = _clamp01(min_active_responsibility)
        self.min_stable_prediction_gain = _clamp01(min_stable_prediction_gain)
        self.min_stable_verified_support = max(1, int(min_stable_verified_support))
        self.contradiction_prune_threshold = max(1, int(contradiction_prune_threshold))
        self.provisional_growth_threshold = _clamp01(provisional_growth_threshold)
        self.responsibility_decay = _clamp01(responsibility_decay)
        self.longevity_increment = _clamp01(longevity_increment)
        self.activity_increment = _clamp01(activity_increment)
        self.routes: MutableMapping[RouteKey, StructuralRouteState] = {}
        self.step = 0

    def register_route(
        self,
        route_key: RouteKey,
        *,
        weight: float = 0.0,
        route_state: str = "provisional",
        responsibility: float = 0.0,
        longevity: float = 0.0,
        prediction_gain_support: float = 0.0,
        contradiction_count: int = 0,
        support_count: int = 0,
        verified_support_count: int = 0,
        last_active_step: Optional[int] = None,
        created_step: Optional[int] = None,
    ) -> None:
        normalized_state = route_state if route_state in ROUTE_STATES else "provisional"
        active_step = self.step if last_active_step is None else int(last_active_step)
        birth_step = self.step if created_step is None else int(created_step)
        self.routes[route_key] = StructuralRouteState(
            weight=float(weight),
            route_state=normalized_state,
            responsibility=_clamp01(responsibility),
            longevity=_clamp01(longevity),
            prediction_gain_support=_clamp01(prediction_gain_support),
            contradiction_count=max(0, int(contradiction_count)),
            support_count=max(0, int(support_count)),
            verified_support_count=max(0, int(verified_support_count)),
            last_active_step=active_step,
            created_step=birth_step,
        )

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            f"{pre}:{post}": state.to_dict()
            for (pre, post), state in sorted(self.routes.items())
        }

    def apply_event(
        self,
        *,
        active_routes: Mapping[RouteKey, float],
        signals: Mapping[str, Any],
        event_memory_support: Optional[Mapping[RouteKey, Mapping[str, Any]]] = None,
        candidate_routes: Optional[Mapping[RouteKey, Mapping[str, Any]]] = None,
        route_contradiction_pressure: Optional[Mapping[RouteKey, float]] = None,
        frozen_evaluation: bool = False,
    ) -> StructuralPlasticityResult:
        self.step += 1
        support_map = event_memory_support or {}
        candidate_map = candidate_routes or {}
        contradiction_map = route_contradiction_pressure or {}

        prediction_error = _clamp01(signals.get("prediction_error", 0.0))
        novelty = _clamp01(signals.get("novelty", 0.0))
        reward = _clamp01(signals.get("reward", 0.0))
        contradiction = _clamp01(signals.get("contradiction", 0.0))
        metabolic_headroom = _clamp01(signals.get("metabolic_headroom", 1.0))
        source_backed = bool(signals.get("source_backed", False))
        abstained = bool(signals.get("abstained", False))
        event_score = max(prediction_error, novelty, reward, contradiction)
        event_triggered = event_score >= self.event_threshold

        decision = "update"
        if frozen_evaluation:
            decision = "freeze_evaluation"
        elif abstained:
            decision = "freeze_abstention"
        elif not source_backed:
            decision = "freeze_unverified_source"
        elif metabolic_headroom < self.metabolic_threshold:
            decision = "freeze_metabolic_budget"
        elif not event_triggered:
            decision = "freeze_no_learning_event"
        update_allowed = decision == "update"

        pruned_routes: List[RouteKey] = []
        grown_routes: List[RouteKey] = []
        stabilized_routes: List[RouteKey] = []
        decaying_routes: List[RouteKey] = []
        actions_taken = 0

        for route_key, state in list(self.routes.items()):
            active_value = _clamp01(abs(float(active_routes.get(route_key, 0.0) or 0.0)))
            route_support = support_map.get(route_key, {})
            prediction_gain_support = _clamp01(
                state.prediction_gain_support * self.responsibility_decay
                + float(route_support.get("prediction_gain_support", 0.0) or 0.0)
            )
            verified_increment = int(bool(route_support.get("verified", False) and source_backed))
            support_increment = int(active_value > 0.0 or bool(route_support))
            route_contradiction = _clamp01(
                contradiction_map.get(route_key, contradiction)
            )
            contradiction_count = state.contradiction_count + int(
                route_contradiction >= self.contradiction_growth_block
            )
            responsibility = _clamp01(
                state.responsibility * self.responsibility_decay
                + (active_value * self.activity_increment)
            )
            longevity = _clamp01(
                state.longevity
                + (self.longevity_increment * max(active_value, float(route_support.get("replay_support", 0.0) or 0.0)))
            )
            next_state = StructuralRouteState(
                weight=state.weight,
                route_state=state.route_state,
                responsibility=responsibility,
                longevity=longevity,
                prediction_gain_support=prediction_gain_support,
                contradiction_count=contradiction_count,
                support_count=state.support_count + support_increment,
                verified_support_count=state.verified_support_count + verified_increment,
                last_active_step=self.step if active_value > 0.0 else state.last_active_step,
                created_step=state.created_step,
            )

            age = self.step - state.created_step
            stale_steps = self.step - next_state.last_active_step

            if (
                next_state.route_state == "provisional"
                and next_state.verified_support_count >= self.min_stable_verified_support
                and next_state.prediction_gain_support >= self.min_stable_prediction_gain
                and route_contradiction < self.contradiction_growth_block
                and update_allowed
                and actions_taken < self.max_rewrites_per_event
            ):
                next_state = StructuralRouteState(
                    **{
                        **next_state.to_dict(),
                        "weight": next_state.weight,
                        "route_state": "stable",
                    }
                )
                stabilized_routes.append(route_key)
                actions_taken += 1
            elif (
                next_state.route_state == "stable"
                and stale_steps >= self.prune_grace_steps
                and next_state.responsibility < self.min_active_responsibility
            ):
                next_state = StructuralRouteState(
                    **{
                        **next_state.to_dict(),
                        "weight": next_state.weight,
                        "route_state": "decaying",
                    }
                )
                decaying_routes.append(route_key)
            elif (
                next_state.route_state == "provisional"
                and age >= self.prune_grace_steps
                and (
                    next_state.responsibility < self.min_active_responsibility
                    or next_state.contradiction_count >= self.contradiction_prune_threshold
                )
            ):
                next_state = StructuralRouteState(
                    **{
                        **next_state.to_dict(),
                        "weight": next_state.weight,
                        "route_state": "decaying",
                    }
                )
                decaying_routes.append(route_key)

            should_prune = (
                state.route_state == "decaying"
                and stale_steps >= self.prune_grace_steps
                and age >= self.prune_grace_steps
                and (
                    next_state.responsibility < self.min_active_responsibility
                    or next_state.contradiction_count >= self.contradiction_prune_threshold
                )
            )
            if should_prune and actions_taken < self.max_rewrites_per_event:
                del self.routes[route_key]
                pruned_routes.append(route_key)
                actions_taken += 1
                continue

            self.routes[route_key] = next_state

        allow_growth = (
            update_allowed
            and actions_taken < self.max_rewrites_per_event
        )
        if allow_growth:
            ordered_candidates = sorted(
                candidate_map.items(),
                key=lambda item: float(item[1].get("coactivation", 0.0) or 0.0)
                + float(item[1].get("prediction_gain_support", 0.0) or 0.0),
                reverse=True,
            )
            for route_key, evidence in ordered_candidates:
                if actions_taken >= self.max_rewrites_per_event:
                    break
                if route_key in self.routes:
                    continue
                growth_score = max(
                    _clamp01(evidence.get("coactivation", 0.0)),
                    _clamp01(evidence.get("prediction_gain_support", 0.0)),
                )
                candidate_contradiction = _clamp01(
                    evidence.get("contradiction_pressure", contradiction)
                )
                if (
                    growth_score < self.provisional_growth_threshold
                    or candidate_contradiction >= self.contradiction_growth_block
                ):
                    continue
                if not bool(evidence.get("verified", False)):
                    continue
                if len(self.routes) >= self.max_total_links:
                    break
                if not self._within_fan_budget(route_key):
                    continue
                self.routes[route_key] = StructuralRouteState(
                    weight=float(evidence.get("weight", 0.0) or 0.0),
                    route_state="provisional",
                    responsibility=_clamp01(evidence.get("responsibility", growth_score)),
                    longevity=_clamp01(evidence.get("longevity", 0.0)),
                    prediction_gain_support=_clamp01(
                        evidence.get("prediction_gain_support", 0.0)
                    ),
                    contradiction_count=0,
                    support_count=1,
                    verified_support_count=1,
                    last_active_step=self.step,
                    created_step=self.step,
                )
                grown_routes.append(route_key)
                actions_taken += 1

        event_cost = (
            len(active_routes)
            + len(candidate_map)
            + len(support_map)
            + len(pruned_routes)
            + len(grown_routes)
            + len(stabilized_routes)
        )
        return StructuralPlasticityResult(
            decision=decision,
            update_allowed=update_allowed,
            event_triggered=event_triggered,
            pruned_routes=tuple(pruned_routes),
            grown_routes=tuple(grown_routes),
            stabilized_routes=tuple(stabilized_routes),
            decaying_routes=tuple(decaying_routes),
            event_cost=event_cost,
            state_budget_units=len(self.routes),
            trace={
                "step": self.step,
                "route_count": len(self.routes),
                "max_total_links": self.max_total_links,
                "max_rewrites_per_event": self.max_rewrites_per_event,
                "actions_taken": actions_taken,
                "snapshot": self.snapshot(),
            },
        )

    def _within_fan_budget(self, route_key: RouteKey) -> bool:
        pre, post = route_key
        fan_out = sum(1 for route_pre, _ in self.routes if route_pre == pre)
        fan_in = sum(1 for _, route_post in self.routes if route_post == post)
        return fan_out < self.max_fan_out and fan_in < self.max_fan_in
