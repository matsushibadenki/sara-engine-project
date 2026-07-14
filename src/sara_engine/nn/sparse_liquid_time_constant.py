"""Bounded event-driven liquid time-constant dynamics for sparse SNN experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class SparseLiquidTrace:
    state: float
    time_constant: float
    spike: bool
    event_cost: int
    update_count: int


class SparseLiquidTimeConstantNeuron:
    """A single bounded liquid neuron using closed-form event updates only."""

    def __init__(
        self,
        *,
        tau: float = 8.0,
        tau_min: float = 2.0,
        tau_max: float = 24.0,
        threshold: float = 0.62,
        max_state: float = 1.0,
        adaptive_threshold: bool = True,
    ) -> None:
        if tau_min <= 0 or tau_max < tau_min:
            raise ValueError("tau bounds must be positive and ordered")
        if not tau_min <= tau <= tau_max:
            raise ValueError("tau must be within tau bounds")
        if threshold <= 0 or max_state <= 0:
            raise ValueError("threshold and max_state must be positive")
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.threshold = float(threshold)
        self.adaptive_threshold = bool(adaptive_threshold)
        self.max_state = float(max_state)
        self.time_constant = float(tau)
        self.state = 0.0

    def reset(self) -> None:
        self.state = 0.0

    def step(self, value: float, gap: int = 1) -> SparseLiquidTrace:
        """Advance only at an event, with one bounded local tau adaptation."""
        if gap < 1:
            raise ValueError("gap must be at least one")
        bounded_value = max(-self.max_state, min(self.max_state, float(value)))
        gap_target = max(self.tau_min, min(self.tau_max, 1.0 + float(gap) * 1.5))
        self.time_constant = max(
            self.tau_min,
            min(self.tau_max, 0.75 * self.time_constant + 0.25 * gap_target),
        )
        threshold = self.threshold
        if self.adaptive_threshold:
            threshold = max(0.45, min(0.8, self.threshold - min(float(gap), 12.0) * 0.015))
        decay = math.exp(-float(gap) / self.time_constant)
        self.state = max(
            -self.max_state,
            min(self.max_state, decay * self.state + (1.0 - decay) * bounded_value),
        )
        spike = self.state >= threshold
        if spike:
            self.state = 0.0
        return SparseLiquidTrace(
            state=round(self.state, 6),
            time_constant=round(self.time_constant, 6),
            spike=spike,
            event_cost=4,
            update_count=1,
        )

    def run(self, events: Iterable[Tuple[int, float]]) -> Tuple[SparseLiquidTrace, ...]:
        return tuple(self.step(value, gap) for gap, value in events)
