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
        max_active_events: int = 64,
        max_state_units: int = 1024,
        max_link_updates_per_call: int = 128,
        homeostatic_clip: float = 1.5,
    ) -> None:
        self.threshold = float(threshold)
        self.feedback_gain = float(feedback_gain)
        self.lateral_gain = float(lateral_gain)
        self.inhibition_gain = float(inhibition_gain)
        self.max_steps = max(1, int(max_steps))
        self.event_budget = max(1, int(event_budget))
        self.max_neighbors_per_event = max(0, int(max_neighbors_per_event))
        self.max_active_events = max(1, int(max_active_events))
        self.max_state_units = max(1, int(max_state_units))
        self.max_link_updates_per_call = max(0, int(max_link_updates_per_call))
        self.homeostatic_clip = max(0.1, float(homeostatic_clip))
        self.local_weights: MutableMapping[Tuple[int, int], float] = defaultdict(float)
        self.event_bias: MutableMapping[int, float] = defaultdict(float)
        self.update_count = 0
        self.last_update_trace: Dict[str, object] = {}

    @staticmethod
    def _bounded_unique(values: Iterable[int], limit: int) -> Tuple[List[int], bool]:
        accepted: List[int] = []
        seen: Set[int] = set()
        overflow = False
        for raw_value in values:
            value = int(raw_value)
            if value in seen:
                continue
            if len(accepted) >= limit:
                overflow = True
                break
            seen.add(value)
            accepted.append(value)
        return sorted(accepted), overflow

    def _bounded_neighbors(
        self,
        event_id: int,
        neighbor_activity: Mapping[int, Iterable[int]],
    ) -> List[int]:
        if self.max_neighbors_per_event == 0:
            return []
        neighbors, _ = self._bounded_unique(
            neighbor_activity.get(int(event_id), []),
            self.max_neighbors_per_event,
        )
        return neighbors

    def state_budget_units(self) -> int:
        return int(len(self.local_weights) + len(self.event_bias))

    def update_local_links(
        self,
        active_events: Iterable[int],
        *,
        learning_rate: float = 0.05,
    ) -> None:
        events, input_overflow = self._bounded_unique(active_events, self.max_active_events)
        bounded_rate = max(0.0, min(float(learning_rate), 1.0))
        link_updates = 0
        state_rejections = 0
        for left in events:
            if left not in self.event_bias and self.state_budget_units() >= self.max_state_units:
                state_rejections += 1
                continue
            self.event_bias[left] = max(
                -self.homeostatic_clip,
                min(self.homeostatic_clip, self.event_bias[left] * 0.98 + bounded_rate),
            )
            for right in events:
                if left == right:
                    continue
                if link_updates >= self.max_link_updates_per_call:
                    break
                key = (left, right)
                if key not in self.local_weights and self.state_budget_units() >= self.max_state_units:
                    state_rejections += 1
                    continue
                updated = self.local_weights[key] * 0.98 + bounded_rate
                self.local_weights[key] = max(-self.homeostatic_clip, min(self.homeostatic_clip, updated))
                link_updates += 1
        self.update_count += 1
        self.last_update_trace = {
            "admitted_event_count": len(events),
            "input_overflow": input_overflow,
            "link_updates": link_updates,
            "state_rejections": state_rejections,
            "state_budget_units": self.state_budget_units(),
            "max_active_events": self.max_active_events,
            "max_state_units": self.max_state_units,
            "max_link_updates_per_call": self.max_link_updates_per_call,
        }

    def gate(
        self,
        *,
        active_event_ids: Sequence[int],
        local_potentials: Optional[Mapping[int, float]] = None,
        recent_output_spikes: Optional[Iterable[int]] = None,
        neighbor_activity: Optional[Mapping[int, Iterable[int]]] = None,
        event_budget: Optional[int] = None,
    ) -> DendriticGateResult:
        budget = self.event_budget if event_budget is None else max(1, int(event_budget))
        admission_limit = min(self.max_active_events, budget)
        active_events, input_overflow = self._bounded_unique(active_event_ids, admission_limit)
        potentials = local_potentials or {}
        recent_output_values, recent_output_overflow = self._bounded_unique(
            recent_output_spikes or [],
            self.max_active_events,
        )
        recent_outputs: Set[int] = set(recent_output_values)
        neighbors_by_event = neighbor_activity or {}
        baseline_events = [
            event_id
            for event_id in active_events
            if float(potentials.get(event_id, 0.0)) >= self.threshold
        ]

        event_cost = len(active_events)
        trace_rows: List[Dict[str, object]] = []
        gated_events: List[int] = []
        convergence_steps = 0

        budget_exhausted = input_overflow
        for _step in range(self.max_steps):
            if budget_exhausted:
                break
            convergence_steps += 1
            gated_events = []
            for event_id in active_events:
                neighbors = self._bounded_neighbors(event_id, neighbors_by_event)
                next_cost = 1 + len(neighbors)
                if event_cost + next_cost > budget:
                    budget_exhausted = True
                    break
                event_cost += next_cost
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

        fallback_used = budget_exhausted
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
                "input_overflow": input_overflow,
                "recent_output_overflow": recent_output_overflow,
                "budget_exhausted": budget_exhausted,
                "trace_rows": trace_rows,
                "max_steps": self.max_steps,
                "update_count": self.update_count,
                "max_active_events": self.max_active_events,
                "max_state_units": self.max_state_units,
                "max_link_updates_per_call": self.max_link_updates_per_call,
                "last_update_trace": dict(self.last_update_trace),
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
