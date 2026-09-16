"""Bounded scalar readout learning from delayed, prediction-bound outcomes.

One instance owns one readout and at most one pending decision. Callers provide
local active routes, not network gradients. The optional residual accumulator is
an experimental ablation, not PC-ALM or a BP approximation. No shared-thread or
durable-state contract is provided. Verification of outcomes belongs to callers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Tuple


def _real(name: str, value: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} is outside its finite range")
    return number


def _integer(name: str, value: int, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _clip(value: float, cap: float) -> float:
    return max(-cap, min(cap, value))


@dataclass(frozen=True)
class LocalOutcomeConfig:
    learning_rate: float = 0.15
    trace_decay: float = 0.99
    max_age: int = 64
    max_routes: int = 256
    max_active: int = 8
    max_delta: float = 0.5
    integral_rate: float = 0.0
    integral_decay: float = 0.999
    integral_cap: float = 1.0
    integral_max_age: int = 4096

    def __post_init__(self) -> None:
        for name in ("learning_rate", "trace_decay", "integral_rate", "integral_decay"):
            _real(name, getattr(self, name), 0.0, 1.0)
        _real("max_delta", self.max_delta, 1e-12, 2.0)
        _real("integral_cap", self.integral_cap, 1e-12, 2.0)
        _integer("max_age", self.max_age, 1, 2**31 - 1)
        _integer("integral_max_age", self.integral_max_age, 1, 2**31 - 1)
        _integer("max_routes", self.max_routes, 1, 4096)
        _integer("max_active", self.max_active, 1, self.max_routes)


@dataclass(frozen=True)
class LocalPrediction:
    sequence: int
    time: int
    score: float
    active: Tuple[Tuple[int, float], ...]


@dataclass(frozen=True)
class LocalUpdate:
    decision: str
    residual: float
    deltas: Tuple[Tuple[int, float], ...]
    route_visits: int


class BoundedLocalOutcomeReadout:
    """Explicit opt-in local delta rule; the default uses no error integrator."""

    def __init__(self, config: Optional[LocalOutcomeConfig] = None) -> None:
        self.config = config or LocalOutcomeConfig()
        self._weights: dict[int, float] = {}
        self._integrals: dict[int, Tuple[float, int]] = {}
        self._pending: Optional[LocalPrediction] = None
        self._time = 0
        self._sequence = 0

    def _check_time(self, time: int) -> None:
        _integer("time", time, 0, 2**53 - 1)
        if time < self._time:
            raise ValueError("time must be monotonic")

    def predict(self, active: Iterable[Tuple[int, float]], *, time: int) -> LocalPrediction:
        self._check_time(time)
        if self._pending is not None:
            raise ValueError("Resolve or discard the pending prediction first")
        if self._sequence == 2**63 - 1:
            raise ValueError("Prediction sequence exhausted")
        # Probe one item past the cap; never materialize unbounded input.
        admitted: dict[int, float] = {}
        for index, (route, strength) in enumerate(active):
            if index >= self.config.max_active:
                raise ValueError("Active route budget exceeded")
            _integer("route", route, 0, 2**63 - 1)
            value = _real("eligibility", strength, 1e-12, 1.0)
            if route in admitted:
                raise ValueError("Duplicate active route")
            admitted[route] = value
        if not admitted:
            raise ValueError("At least one active route is required")
        additions = sum(route not in self._weights for route in admitted)
        if len(self._weights) + additions > self.config.max_routes:
            raise ValueError("Readout route budget exceeded")
        ordered = tuple(sorted(admitted.items()))
        total = sum(value for _, value in ordered)
        score = sum(self._weights.get(route, 0.0) * value for route, value in ordered) / total
        # Commit only after full validation. New routes are created on feedback.
        receipt = LocalPrediction(self._sequence + 1, time, _clip(score, 1.0), ordered)
        self._pending = receipt
        self._sequence += 1
        self._time = time
        return receipt

    def _check_receipt(self, prediction: LocalPrediction, time: int) -> None:
        self._check_time(time)
        # Object identity rejects copied, forged and cross-instance receipts.
        if self._pending is None or prediction is not self._pending:
            raise ValueError("Prediction receipt is not pending in this readout")

    def discard(self, prediction: LocalPrediction, *, time: int) -> None:
        self._check_receipt(prediction, time)
        self._pending = None
        self._time = time

    def observe(self, prediction: LocalPrediction, outcome: float, *, time: int) -> LocalUpdate:
        self._check_receipt(prediction, time)
        target = _real("outcome", outcome, -1.0, 1.0)
        age = time - prediction.time
        residual = target - prediction.score
        if age > self.config.max_age:
            self.discard(prediction, time=time)
            return LocalUpdate("expired", residual, (), 0)
        if self.config.learning_rate == 0.0:
            self.discard(prediction, time=time)
            return LocalUpdate("frozen", residual, (), 0)
        attenuation = self.config.trace_decay ** age
        total = sum(value for _, value in prediction.active)
        updates = []
        for route, eligibility in prediction.active:
            integral = 0.0
            if self.config.integral_rate:
                previous, last_time = self._integrals.get(route, (0.0, time))
                # Old credit cannot survive the configured feedback horizon.
                if time - last_time > self.config.integral_max_age:
                    previous = 0.0
                integral = _clip(
                    previous * self.config.integral_decay ** (time - last_time)
                    + self.config.integral_rate * residual,
                    self.config.integral_cap,
                )
                self._integrals[route] = (integral, time)
            # Match the instantaneous gain of the no-integral control.
            signal = (residual + integral) / (1.0 + self.config.integral_rate)
            delta = _clip(self.config.learning_rate * attenuation * eligibility / total * signal, self.config.max_delta)
            old = self._weights.get(route, 0.0)
            new = _clip(old + delta, 1.0)
            self._weights[route] = new
            updates.append((route, new - old))
        self.discard(prediction, time=time)
        return LocalUpdate("updated", residual, tuple(updates), len(updates))

    def snapshot(self) -> dict:
        """Return a bounded diagnostic copy, never a mutable state reference."""
        return {
            "weights": dict(self._weights),
            "integrals": dict(self._integrals),
            "pending": self._pending,
            "time": self._time,
            "sequence": self._sequence,
        }

    def get_status(self, lang: str = "en") -> str:
        messages = {
            "en": "Local readout: {routes} routes, {pending} pending outcome",
            "ja": "局所読み出し: {routes}経路、結果待ち{pending}件",
            "zh-CN": "局部读出：{routes}条路径，{pending}个待反馈预测",
        }
        return messages.get(lang, messages["en"]).format(routes=len(self._weights), pending=int(self._pending is not None))
