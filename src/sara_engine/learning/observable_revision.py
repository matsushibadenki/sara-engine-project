"""Bounded local rule selection driven by an explicit revision notification.

The notification is supplied by a caller and contains no label or route. This
module does not detect change or establish that the notification is truthful.
It uses residual updates for a bounded horizon, then returns to signed constant
updates for stable repetition. No matrix, global gradient, or GPU is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Tuple


def _number(name: str, value: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} is outside its finite range")
    return number


def _int(name: str, value: int, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _clip(value: float, cap: float) -> float:
    return max(-cap, min(cap, value))


@dataclass(frozen=True)
class ObservableRevisionConfig:
    learning_rate: float = 0.15
    trace_decay: float = 0.99
    max_feedback_age: int = 64
    adaptation_horizon: int = 1152
    max_routes: int = 256
    max_active: int = 8
    max_delta: float = 0.5

    def __post_init__(self) -> None:
        _number("learning_rate", self.learning_rate, 0.0, 1.0)
        _number("trace_decay", self.trace_decay, 0.0, 1.0)
        _number("max_delta", self.max_delta, 1e-12, 2.0)
        _int("max_feedback_age", self.max_feedback_age, 1, 2**31 - 1)
        _int("adaptation_horizon", self.adaptation_horizon, 1, 2**31 - 1)
        _int("max_routes", self.max_routes, 1, 4096)
        _int("max_active", self.max_active, 1, self.max_routes)


@dataclass(frozen=True)
class RevisionPrediction:
    sequence: int
    time: int
    revision: int
    mode: str
    score: float
    active: Tuple[Tuple[int, float], ...]


@dataclass(frozen=True)
class RevisionUpdate:
    decision: str
    mode: str
    residual: float
    deltas: Tuple[Tuple[int, float], ...]
    adaptive_remaining: int


class BoundedObservableRevisionReadout:
    """Single-owner local readout with explicit, monotonic revision signals."""

    def __init__(self, config: Optional[ObservableRevisionConfig] = None, *, initial_revision: int = 1) -> None:
        self.config = config or ObservableRevisionConfig()
        self._revision = _int("initial_revision", initial_revision, 0, 2**63 - 1)
        self._weights: dict[int, float] = {}
        self._pending: Optional[RevisionPrediction] = None
        self._adaptive_remaining = 0
        self._sequence = 0
        self._time = 0

    def _check_time(self, time: int) -> None:
        _int("time", time, 0, 2**53 - 1)
        if time < self._time:
            raise ValueError("time must be monotonic")

    def notify_revision(self, revision: int, *, time: int) -> None:
        self._check_time(time)
        revision = _int("revision", revision, 0, 2**63 - 1)
        if self._pending is not None:
            raise ValueError("Resolve or discard the pending prediction before revision")
        if revision <= self._revision:
            raise ValueError("revision must increase strictly")
        self._revision = revision
        self._adaptive_remaining = self.config.adaptation_horizon
        self._time = time

    def predict(self, active: Iterable[Tuple[int, float]], *, time: int) -> RevisionPrediction:
        self._check_time(time)
        if self._pending is not None:
            raise ValueError("Resolve or discard the pending prediction first")
        if self._sequence == 2**63 - 1:
            raise ValueError("Prediction sequence exhausted")
        admitted: dict[int, float] = {}
        for index, item in enumerate(active):
            if index >= self.config.max_active:
                raise ValueError("Active route budget exceeded")
            try:
                route, strength = item
            except (TypeError, ValueError) as exc:
                raise ValueError("Each active route must be a pair") from exc
            route = _int("route", route, 0, 2**63 - 1)
            strength = _number("eligibility", strength, 1e-12, 1.0)
            if route in admitted:
                raise ValueError("Duplicate active route")
            admitted[route] = strength
        if not admitted:
            raise ValueError("At least one active route is required")
        additions = sum(route not in self._weights for route in admitted)
        if len(self._weights) + additions > self.config.max_routes:
            raise ValueError("Readout route budget exceeded")
        ordered = tuple(sorted(admitted.items()))
        total = sum(value for _, value in ordered)
        score = sum(self._weights.get(route, 0.0) * value for route, value in ordered) / total
        mode = "residual" if self._adaptive_remaining > 0 else "three_factor"
        receipt = RevisionPrediction(self._sequence + 1, time, self._revision, mode, _clip(score, 1.0), ordered)
        self._pending = receipt
        self._sequence += 1
        self._time = time
        return receipt

    def _validate_receipt(self, prediction: RevisionPrediction, time: int) -> None:
        self._check_time(time)
        if self._pending is None or prediction is not self._pending:
            raise ValueError("Prediction receipt is not pending in this readout")

    def discard(self, prediction: RevisionPrediction, *, time: int) -> None:
        self._validate_receipt(prediction, time)
        self._pending = None
        self._time = time

    def observe(self, prediction: RevisionPrediction, outcome: float, *, time: int) -> RevisionUpdate:
        self._validate_receipt(prediction, time)
        target = _number("outcome", outcome, -1.0, 1.0)
        age = time - prediction.time
        residual = target - prediction.score
        if age > self.config.max_feedback_age:
            self.discard(prediction, time=time)
            return RevisionUpdate("expired", prediction.mode, residual, (), self._adaptive_remaining)
        attenuation = self.config.trace_decay ** age
        total = sum(value for _, value in prediction.active)
        signal = residual if prediction.mode == "residual" else target
        deltas = []
        for route, eligibility in prediction.active:
            delta = _clip(self.config.learning_rate * attenuation * eligibility / total * signal, self.config.max_delta)
            previous = self._weights.get(route, 0.0)
            current = _clip(previous + delta, 1.0)
            self._weights[route] = current
            deltas.append((route, current - previous))
        if prediction.mode == "residual":
            self._adaptive_remaining -= 1
        self.discard(prediction, time=time)
        return RevisionUpdate("updated", prediction.mode, residual, tuple(deltas), self._adaptive_remaining)

    def snapshot(self) -> dict:
        return {
            "weights": dict(self._weights), "pending": self._pending,
            "revision": self._revision, "adaptive_remaining": self._adaptive_remaining,
            "sequence": self._sequence, "time": self._time,
        }

    def get_status(self, lang: str = "en") -> str:
        messages = {
            "en": "Revision readout: revision {revision}, adaptive decisions {remaining}",
            "ja": "改訂読み出し: 改訂{revision}、適応判断残り{remaining}件",
            "zh-CN": "修订读出：修订{revision}，剩余自适应决策{remaining}次",
        }
        return messages.get(lang, messages["en"]).format(revision=self._revision, remaining=self._adaptive_remaining)
