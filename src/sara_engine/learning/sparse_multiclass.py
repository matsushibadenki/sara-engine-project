"""Bounded sparse mistake-driven multiclass local learning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional


@dataclass(frozen=True)
class SparseMulticlassConfig:
    learning_rate: float = 0.30
    weight_cap: float = 2.0
    max_routes: int = 4096
    max_active: int = 8
    max_classes: int = 24

    def __post_init__(self) -> None:
        if not 0.0 <= self.learning_rate <= 1.0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and in [0, 1]")
        if not 0.0 < self.weight_cap <= 16.0 or not math.isfinite(self.weight_cap):
            raise ValueError("weight_cap must be finite and positive")
        for name in ("max_routes", "max_active", "max_classes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_active > self.max_routes:
            raise ValueError("max_active cannot exceed max_routes")


@dataclass(frozen=True)
class SparseClassPrediction:
    sequence: int
    predicted: str
    scores: tuple[tuple[str, float], ...]
    active: tuple[int, ...]


@dataclass(frozen=True)
class SparseClassUpdate:
    decision: str
    updates: int


class BoundedSparseMulticlassReadout:
    def __init__(self, labels: Iterable[str], config: Optional[SparseMulticlassConfig] = None) -> None:
        self.config = config or SparseMulticlassConfig()
        self.labels = tuple(sorted(set(labels)))
        if not self.labels or len(self.labels) > self.config.max_classes:
            raise ValueError("Class vocabulary is empty or exceeds the cap")
        self._weights: dict[tuple[int, str], float] = {}
        self._routes: set[int] = set()
        self._pending: Optional[SparseClassPrediction] = None
        self._sequence = 0

    def predict(self, active: Iterable[int]) -> SparseClassPrediction:
        if self._pending is not None:
            raise ValueError("Resolve the pending prediction first")
        admitted = []
        seen = set()
        for index, route in enumerate(active):
            if index >= self.config.max_active:
                raise ValueError("Active route budget exceeded")
            if type(route) is not int or route < 0 or route in seen:
                raise ValueError("Active routes must be unique non-negative integers")
            seen.add(route); admitted.append(route)
        if not admitted:
            raise ValueError("At least one active route is required")
        if len(self._routes | seen) > self.config.max_routes:
            raise ValueError("Route vocabulary exceeded")
        scores = tuple((label, sum(self._weights.get((route, label), 0.0) for route in admitted))
                       for label in self.labels)
        predicted = max(scores, key=lambda item: (item[1], -self.labels.index(item[0])))[0]
        receipt = SparseClassPrediction(self._sequence + 1, predicted, scores, tuple(admitted))
        self._pending = receipt; self._sequence += 1
        return receipt

    def observe(self, prediction: SparseClassPrediction, target: str) -> SparseClassUpdate:
        if prediction is not self._pending:
            raise ValueError("Prediction receipt is not pending")
        if target not in self.labels:
            raise ValueError("Unknown target class")
        self._routes.update(prediction.active)
        updates = 0
        if target != prediction.predicted and self.config.learning_rate > 0.0:
            delta = self.config.learning_rate / len(prediction.active)
            for route in prediction.active:
                for label, sign in ((target, 1.0), (prediction.predicted, -1.0)):
                    key = (route, label)
                    self._weights[key] = max(-self.config.weight_cap,
                                             min(self.config.weight_cap, self._weights.get(key, 0.0) + sign * delta))
                    updates += 1
        self._pending = None
        return SparseClassUpdate("correct" if target == prediction.predicted else "updated", updates)

    def probabilities(self, prediction: SparseClassPrediction) -> dict[str, float]:
        maximum = max(score for _, score in prediction.scores)
        values = {label: math.exp(score - maximum) for label, score in prediction.scores}
        total = sum(values.values())
        return {label: value / total for label, value in values.items()}

    def snapshot(self) -> dict:
        return {"weights": dict(self._weights), "routes": set(self._routes),
                "pending": self._pending, "sequence": self._sequence}


__all__ = ["BoundedSparseMulticlassReadout", "SparseMulticlassConfig", "SparseClassPrediction", "SparseClassUpdate"]
