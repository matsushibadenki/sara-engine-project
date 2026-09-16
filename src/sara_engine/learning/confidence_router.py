"""Bounded confidence routing between a count model and a local residual."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True, order=True)
class ConfidenceRouteConfig:
    base_probability_cap: float
    local_score_margin: float
    minimum_base_support: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_probability_cap) or not 0.0 <= self.base_probability_cap <= 1.0:
            raise ValueError("base_probability_cap must be finite and in [0, 1]")
        if not math.isfinite(self.local_score_margin) or self.local_score_margin < 0.0:
            raise ValueError("local_score_margin must be finite and non-negative")
        if type(self.minimum_base_support) is not int or self.minimum_base_support < 0:
            raise ValueError("minimum_base_support must be a non-negative integer")


@dataclass(frozen=True)
class ConfidenceRouteDecision:
    predicted: str
    probabilities: tuple[tuple[str, float], ...]
    overridden: bool
    local_margin: float


def route_prediction(
    *,
    labels: Sequence[str],
    base_predicted: str,
    base_probabilities: Mapping[str, float],
    base_support: int,
    local_predicted: str,
    local_scores: Mapping[str, float],
    config: ConfidenceRouteConfig,
) -> ConfidenceRouteDecision:
    """Route one prediction with fixed, scalar and auditable conditions."""
    ordered = tuple(labels)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("labels must be unique and non-empty")
    if base_predicted not in ordered or local_predicted not in ordered:
        raise ValueError("predictions must belong to labels")
    if type(base_support) is not int or base_support < 0:
        raise ValueError("base_support must be a non-negative integer")
    if set(base_probabilities) != set(ordered) or set(local_scores) != set(ordered):
        raise ValueError("probabilities and scores must exactly cover labels")
    probabilities = {label: float(base_probabilities[label]) for label in ordered}
    scores = {label: float(local_scores[label]) for label in ordered}
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities.values()):
        raise ValueError("base probabilities must be finite and non-negative")
    if abs(sum(probabilities.values()) - 1.0) > 1e-9:
        raise ValueError("base probabilities must sum to one")
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("local scores must be finite")
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else math.inf
    override = (
        local_predicted != base_predicted
        and probabilities[base_predicted] <= config.base_probability_cap
        and margin >= config.local_score_margin
        and base_support >= config.minimum_base_support
    )
    predicted = local_predicted if override else base_predicted
    if override:
        probabilities[base_predicted], probabilities[local_predicted] = (
            probabilities[local_predicted], probabilities[base_predicted]
        )
    return ConfidenceRouteDecision(
        predicted=predicted,
        probabilities=tuple((label, probabilities[label]) for label in ordered),
        overridden=override,
        local_margin=margin,
    )


__all__ = ["ConfidenceRouteConfig", "ConfidenceRouteDecision", "route_prediction"]
