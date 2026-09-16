"""Outcome-blind score normalization for bounded local override routing."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

@dataclass(frozen=True)
class NormalizedRouteDecision:
    predicted: str
    probabilities: tuple[tuple[str, float], ...]
    overridden: bool
    score: float

def normalized_override_score(*, base_predicted: str, base_probabilities: Mapping[str, float], base_support: int,
        local_predicted: str, local_scores: Mapping[str, float], minimum_base_support: int = 16) -> float | None:
    if base_support < minimum_base_support or local_predicted == base_predicted:
        return None
    values = sorted((float(value) for value in local_scores.values()), reverse=True)
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("local scores require at least two finite values")
    confidence = float(base_probabilities[base_predicted])
    if not math.isfinite(confidence) or confidence <= 0.0:
        raise ValueError("base confidence must be finite and positive")
    return (values[0] - values[1]) / confidence

def calibrate_override_threshold(scores: Sequence[float | None], target_fraction: float = .10) -> float:
    if not 0.0 < target_fraction < 1.0 or not scores:
        raise ValueError("target_fraction and scores must be non-empty and bounded")
    eligible = sorted((score for score in scores if score is not None), reverse=True)
    budget = math.ceil(len(scores) * target_fraction)
    return eligible[budget - 1] if len(eligible) >= budget else math.inf

def route_normalized(*, labels: Sequence[str], base_predicted: str, base_probabilities: Mapping[str, float],
        base_support: int, local_predicted: str, local_scores: Mapping[str, float], threshold: float,
        minimum_base_support: int = 16) -> NormalizedRouteDecision:
    ordered = tuple(labels); probabilities = {label: float(base_probabilities[label]) for label in ordered}
    score = normalized_override_score(base_predicted=base_predicted, base_probabilities=probabilities,
        base_support=base_support, local_predicted=local_predicted, local_scores=local_scores,
        minimum_base_support=minimum_base_support)
    override = score is not None and score >= threshold
    predicted = local_predicted if override else base_predicted
    if override:
        probabilities[base_predicted], probabilities[local_predicted] = probabilities[local_predicted], probabilities[base_predicted]
    return NormalizedRouteDecision(predicted, tuple((label, probabilities[label]) for label in ordered), override, score or 0.0)

__all__ = ["NormalizedRouteDecision", "calibrate_override_threshold", "normalized_override_score", "route_normalized"]
