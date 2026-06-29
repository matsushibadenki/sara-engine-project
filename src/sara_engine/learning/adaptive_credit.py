from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple


RouteKey = Tuple[int, int]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _bucketize(value: float, buckets: Sequence[float]) -> float:
    if not buckets:
        return value
    bounded = _clamp(value, 0.0, 1.0)
    for candidate in buckets:
        if candidate >= bounded:
            return candidate
    return buckets[-1]


@dataclass(frozen=True)
class AdaptiveCreditRouteState:
    weight: float = 0.0
    eligibility: float = 0.0
    responsibility: float = 0.0
    confidence: float = 0.0
    longevity: float = 0.0
    update_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight": round(self.weight, 6),
            "eligibility": round(self.eligibility, 6),
            "responsibility": round(self.responsibility, 6),
            "confidence": round(self.confidence, 6),
            "longevity": round(self.longevity, 6),
            "update_count": self.update_count,
        }


@dataclass(frozen=True)
class AdaptiveCreditResult:
    update_allowed: bool
    decision: str
    event_triggered: bool
    touched_route_count: int
    updated_route_count: int
    skipped_by_region_count: int
    event_cost: int
    state_budget_units: int
    quantized_credit_mode: bool
    updates: Dict[RouteKey, Dict[str, float]]
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_allowed": self.update_allowed,
            "decision": self.decision,
            "event_triggered": self.event_triggered,
            "touched_route_count": self.touched_route_count,
            "updated_route_count": self.updated_route_count,
            "skipped_by_region_count": self.skipped_by_region_count,
            "event_cost": self.event_cost,
            "state_budget_units": self.state_budget_units,
            "quantized_credit_mode": self.quantized_credit_mode,
            "updates": {
                f"{pre}:{post}": {key: round(value, 6) for key, value in payload.items()}
                for (pre, post), payload in sorted(self.updates.items())
            },
            "trace": dict(self.trace),
        }


class AdaptiveCreditField:
    """Sparse event-driven local credit assignment for delayed learning signals."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.12,
        eligibility_decay: float = 0.85,
        responsibility_decay: float = 0.9,
        confidence_decay: float = 0.96,
        longevity_increment: float = 0.08,
        activity_increment: float = 0.25,
        event_threshold: float = 0.35,
        region_threshold: float = 0.2,
        weight_clip: float = 1.5,
        max_routes: int = 256,
        quantize_credit: bool = False,
        credit_buckets: Sequence[float] = (0.0, 0.33, 0.66, 1.0),
    ) -> None:
        self.learning_rate = _clamp(learning_rate, 0.0, 1.0)
        self.eligibility_decay = _clamp(eligibility_decay, 0.0, 1.0)
        self.responsibility_decay = _clamp(responsibility_decay, 0.0, 1.0)
        self.confidence_decay = _clamp(confidence_decay, 0.0, 1.0)
        self.longevity_increment = _clamp(longevity_increment, 0.0, 1.0)
        self.activity_increment = _clamp(activity_increment, 0.0, 1.0)
        self.event_threshold = _clamp(event_threshold, 0.0, 1.0)
        self.region_threshold = _clamp(region_threshold, 0.0, 1.0)
        self.weight_clip = max(0.1, float(weight_clip))
        self.max_routes = max(1, int(max_routes))
        self.quantize_credit = bool(quantize_credit)
        self.credit_buckets = tuple(
            sorted({_clamp(float(bucket), 0.0, 1.0) for bucket in credit_buckets})
        )
        self.routes: MutableMapping[RouteKey, AdaptiveCreditRouteState] = {}
        self.update_count = 0
        self.freeze_count = 0
        self.last_event_cost = 0

    def apply(
        self,
        *,
        active_routes: Mapping[RouteKey, float],
        signals: Mapping[str, Any],
        route_regions: Optional[Mapping[RouteKey, str]] = None,
        region_credit: Optional[Mapping[str, float]] = None,
    ) -> AdaptiveCreditResult:
        normalized_signals = {
            "prediction_error": _clamp(float(signals.get("prediction_error", 0.0) or 0.0), 0.0, 1.0),
            "novelty": _clamp(float(signals.get("novelty", 0.0) or 0.0), 0.0, 1.0),
            "reward": _clamp(float(signals.get("reward", 0.0) or 0.0), 0.0, 1.0),
            "verifier_disagreement": _clamp(
                float(signals.get("verifier_disagreement", 0.0) or 0.0),
                0.0,
                1.0,
            ),
            "contradiction": _clamp(float(signals.get("contradiction", 0.0) or 0.0), 0.0, 1.0),
            "metabolic_headroom": _clamp(
                float(signals.get("metabolic_headroom", 1.0) or 0.0),
                0.0,
                1.0,
            ),
        }
        source_backed = bool(signals.get("source_backed", False))
        abstained = bool(signals.get("abstained", False))
        event_score = max(
            normalized_signals["prediction_error"],
            normalized_signals["novelty"],
            normalized_signals["reward"],
            normalized_signals["verifier_disagreement"],
            normalized_signals["contradiction"],
        )
        event_triggered = event_score >= self.event_threshold
        decision = "update"
        if abstained:
            decision = "freeze_abstention"
        elif not source_backed:
            decision = "freeze_unverified_source"
        elif normalized_signals["contradiction"] >= self.event_threshold:
            decision = "freeze_contradiction"
        elif normalized_signals["metabolic_headroom"] < self.region_threshold:
            decision = "freeze_metabolic_budget"
        elif not event_triggered:
            decision = "freeze_no_learning_event"

        update_allowed = decision == "update"
        updates: Dict[RouteKey, Dict[str, float]] = {}
        touched_route_count = 0
        updated_route_count = 0
        skipped_by_region_count = 0
        route_regions = route_regions or {}
        region_credit = region_credit or {}
        bounded_region_credit = {
            str(key): _clamp(float(value), 0.0, 1.0)
            for key, value in region_credit.items()
        }

        if update_allowed:
            ordered_active = sorted(
                active_routes.items(),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )
            for route_key, activity in ordered_active:
                if route_key not in self.routes and len(self.routes) >= self.max_routes:
                    continue
                touched_route_count += 1
                region = str(route_regions.get(route_key, "global"))
                region_gate = bounded_region_credit.get(region, 1.0)
                if region_gate < self.region_threshold:
                    skipped_by_region_count += 1
                    continue
                state = self.routes.get(route_key, AdaptiveCreditRouteState())
                active_value = _clamp(abs(float(activity)), 0.0, 1.0)
                eligibility = _clamp(
                    state.eligibility * self.eligibility_decay
                    + (active_value * self.activity_increment),
                    0.0,
                    1.0,
                )
                local_credit = _clamp(
                    (
                        normalized_signals["prediction_error"]
                        + normalized_signals["novelty"]
                        + normalized_signals["reward"]
                        + normalized_signals["verifier_disagreement"]
                    )
                    / 4.0
                    * region_gate
                    * normalized_signals["metabolic_headroom"],
                    0.0,
                    1.0,
                )
                responsibility = _clamp(
                    state.responsibility * self.responsibility_decay
                    + (eligibility * local_credit),
                    0.0,
                    1.0,
                )
                confidence = _clamp(
                    state.confidence * self.confidence_decay
                    + ((1.0 - normalized_signals["contradiction"]) * region_gate * 0.5),
                    0.0,
                    1.0,
                )
                longevity = _clamp(
                    state.longevity + (self.longevity_increment * local_credit),
                    0.0,
                    1.0,
                )
                if self.quantize_credit:
                    responsibility = _bucketize(responsibility, self.credit_buckets)
                    confidence = _bucketize(confidence, self.credit_buckets)
                    longevity = _bucketize(longevity, self.credit_buckets)
                delta = self.learning_rate * eligibility * responsibility
                updated_weight = _clamp(
                    state.weight + delta,
                    -self.weight_clip,
                    self.weight_clip,
                )
                next_state = AdaptiveCreditRouteState(
                    weight=updated_weight,
                    eligibility=eligibility,
                    responsibility=responsibility,
                    confidence=confidence,
                    longevity=longevity,
                    update_count=state.update_count + 1,
                )
                self.routes[route_key] = next_state
                updated_route_count += 1
                updates[route_key] = {
                    "delta": delta,
                    "eligibility": eligibility,
                    "responsibility": responsibility,
                    "confidence": confidence,
                    "longevity": longevity,
                    "region_gate": region_gate,
                }
            self.update_count += 1
        else:
            self.freeze_count += 1

        event_cost = len(normalized_signals) + touched_route_count + skipped_by_region_count
        self.last_event_cost = event_cost
        return AdaptiveCreditResult(
            update_allowed=update_allowed,
            decision=decision,
            event_triggered=event_triggered,
            touched_route_count=touched_route_count,
            updated_route_count=updated_route_count,
            skipped_by_region_count=skipped_by_region_count,
            event_cost=event_cost,
            state_budget_units=len(self.routes),
            quantized_credit_mode=self.quantize_credit,
            updates=updates,
            trace={
                "signals": normalized_signals,
                "event_score": round(event_score, 6),
                "source_backed": source_backed,
                "abstained": abstained,
                "region_credit": bounded_region_credit,
                "update_count": self.update_count,
                "freeze_count": self.freeze_count,
            },
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-adaptive-credit-field-state-v1",
            "quantized_credit_mode": self.quantize_credit,
            "route_count": len(self.routes),
            "update_count": self.update_count,
            "freeze_count": self.freeze_count,
            "last_event_cost": self.last_event_cost,
            "routes": {
                f"{pre}:{post}": state.to_dict()
                for (pre, post), state in sorted(self.routes.items())
            },
        }


def summarize_event_memory_credit(
    route_states: Iterable[Mapping[str, Any] | AdaptiveCreditRouteState],
    *,
    max_routes: int = 4,
) -> Dict[str, float]:
    """Compress sparse route-local credit into a bounded Event Memory summary."""
    normalized: list[tuple[float, float, float]] = []
    for raw in route_states:
        if isinstance(raw, AdaptiveCreditRouteState):
            responsibility = raw.responsibility
            confidence = raw.confidence
            longevity = raw.longevity
        elif isinstance(raw, Mapping):
            responsibility = float(raw.get("responsibility", 0.0) or 0.0)
            confidence = float(raw.get("confidence", 0.0) or 0.0)
            longevity = float(raw.get("longevity", 0.0) or 0.0)
        else:
            continue
        normalized.append(
            (
                _clamp(responsibility, 0.0, 1.0),
                _clamp(confidence, 0.0, 1.0),
                _clamp(longevity, 0.0, 1.0),
            )
        )
    if not normalized:
        return {
            "credit_score": 0.0,
            "credit_responsibility": 0.0,
            "credit_confidence": 0.0,
            "credit_longevity": 0.0,
            "credit_route_count": 0.0,
        }
    ranked = sorted(
        normalized,
        key=lambda item: (item[0], item[1], item[2]),
        reverse=True,
    )[: max(1, int(max_routes))]
    route_count = len(ranked)
    responsibility = sum(item[0] for item in ranked) / float(route_count)
    confidence = sum(item[1] for item in ranked) / float(route_count)
    longevity = sum(item[2] for item in ranked) / float(route_count)
    credit_score = _clamp(
        0.45 * responsibility + 0.30 * confidence + 0.25 * longevity,
        0.0,
        1.0,
    )
    return {
        "credit_score": round(credit_score, 6),
        "credit_responsibility": round(responsibility, 6),
        "credit_confidence": round(confidence, 6),
        "credit_longevity": round(longevity, 6),
        "credit_route_count": float(route_count),
    }
