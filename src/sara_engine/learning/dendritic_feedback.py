from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class DendriticGateResult:
    gated_events: List[int]
    fallback_used: bool
    convergence_steps: int
    event_cost: int
    state_budget_units: int
    trace: Dict[str, object]


class SparseDendriticFeedbackGate:
    """Bounded sparse dendritic-style gate with local feedback only."""

    def __init__(
        self,
        *,
        threshold: float = 1.0,
        feedback_gain: float = 0.3,
        lateral_gain: float = 0.2,
        inhibition_gain: float = 0.15,
        max_steps: int = 1,
        event_budget: int = 128,
        max_neighbors_per_event: int = 8,
        homeostatic_clip: float = 1.5,
    ) -> None:
        self.threshold = float(threshold)
        self.feedback_gain = float(feedback_gain)
        self.lateral_gain = float(lateral_gain)
        self.inhibition_gain = float(inhibition_gain)
        self.max_steps = max(1, int(max_steps))
        self.event_budget = max(1, int(event_budget))
        self.max_neighbors_per_event = max(0, int(max_neighbors_per_event))
        self.homeostatic_clip = max(0.1, float(homeostatic_clip))
        self.local_weights: MutableMapping[Tuple[int, int], float] = defaultdict(float)
        self.event_bias: MutableMapping[int, float] = defaultdict(float)
        self.update_count = 0

    def _bounded_neighbors(
        self,
        event_id: int,
        neighbor_activity: Mapping[int, Iterable[int]],
    ) -> List[int]:
        neighbors = [int(item) for item in neighbor_activity.get(int(event_id), [])]
        return sorted(set(neighbors))[: self.max_neighbors_per_event]

    def state_budget_units(self) -> int:
        return int(len(self.local_weights) + len(self.event_bias))

    def update_local_links(
        self,
        active_events: Iterable[int],
        *,
        learning_rate: float = 0.05,
    ) -> None:
        events = sorted(set(int(item) for item in active_events))
        bounded_rate = max(0.0, min(float(learning_rate), 1.0))
        for left in events:
            self.event_bias[left] = max(
                -self.homeostatic_clip,
                min(self.homeostatic_clip, self.event_bias[left] * 0.98 + bounded_rate),
            )
            for right in events:
                if left == right:
                    continue
                key = (left, right)
                updated = self.local_weights[key] * 0.98 + bounded_rate
                self.local_weights[key] = max(-self.homeostatic_clip, min(self.homeostatic_clip, updated))
        self.update_count += 1

    def gate(
        self,
        *,
        active_event_ids: Sequence[int],
        local_potentials: Optional[Mapping[int, float]] = None,
        recent_output_spikes: Optional[Iterable[int]] = None,
        neighbor_activity: Optional[Mapping[int, Iterable[int]]] = None,
        event_budget: Optional[int] = None,
    ) -> DendriticGateResult:
        active_events = sorted(set(int(item) for item in active_event_ids))
        potentials = dict(local_potentials or {})
        recent_outputs: Set[int] = set(int(item) for item in (recent_output_spikes or []))
        neighbors_by_event = dict(neighbor_activity or {})
        budget = self.event_budget if event_budget is None else max(1, int(event_budget))
        baseline_events = [
            event_id
            for event_id in active_events
            if float(potentials.get(event_id, 0.0)) >= self.threshold
        ]

        event_cost = len(active_events)
        trace_rows: List[Dict[str, object]] = []
        gated_events: List[int] = []
        convergence_steps = 0

        for _step in range(self.max_steps):
            convergence_steps += 1
            gated_events = []
            for event_id in active_events:
                neighbors = self._bounded_neighbors(event_id, neighbors_by_event)
                event_cost += 1 + len(neighbors)
                base = float(potentials.get(event_id, 0.0))
                feedback = self.feedback_gain if event_id in recent_outputs else 0.0
                lateral = 0.0
                for neighbor_id in neighbors:
                    lateral += self.local_weights.get((neighbor_id, event_id), 0.0)
                    if neighbor_id in active_events:
                        lateral += self.lateral_gain
                inhibition = self.inhibition_gain * max(0, len(neighbors) - 1)
                bias = self.event_bias.get(event_id, 0.0)
                adjusted = base + feedback + lateral + bias - inhibition
                passed = adjusted >= self.threshold
                if passed:
                    gated_events.append(event_id)
                trace_rows.append(
                    {
                        "event_id": event_id,
                        "base_potential": round(base, 6),
                        "feedback": round(feedback, 6),
                        "lateral": round(lateral, 6),
                        "bias": round(bias, 6),
                        "inhibition": round(inhibition, 6),
                        "adjusted_potential": round(adjusted, 6),
                        "passed": passed,
                        "neighbor_count": len(neighbors),
                    }
                )
            break

        fallback_used = event_cost > budget
        if fallback_used:
            gated_events = baseline_events

        return DendriticGateResult(
            gated_events=sorted(gated_events),
            fallback_used=fallback_used,
            convergence_steps=convergence_steps,
            event_cost=event_cost,
            state_budget_units=self.state_budget_units(),
            trace={
                "baseline_events": sorted(baseline_events),
                "event_budget": budget,
                "trace_rows": trace_rows,
                "max_steps": self.max_steps,
                "update_count": self.update_count,
            },
        )


def precision_at_expected(predicted: Iterable[int], expected: Iterable[int]) -> float:
    predicted_set = set(int(item) for item in predicted)
    expected_set = set(int(item) for item in expected)
    if not predicted_set and not expected_set:
        return 1.0
    if not predicted_set:
        return 0.0
    return float(len(predicted_set.intersection(expected_set))) / float(len(predicted_set))
