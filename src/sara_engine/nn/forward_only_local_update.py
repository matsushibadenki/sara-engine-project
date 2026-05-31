# Directory Path: src/sara_engine/nn/forward_only_local_update.py
# English Title: Forward-Only Local Update Trace
# Purpose/Content: Bounded sparse eligibility trace for local online learning probes without BPTT.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _event_ids(events: Sequence[Any]) -> List[int]:
    ids: List[int] = []
    for event in events:
        if hasattr(event, "spike_id"):
            ids.append(int(event.spike_id))
        elif isinstance(event, Mapping) and "spike_id" in event:
            ids.append(int(event["spike_id"]))
        else:
            ids.append(int(event))
    return sorted(set(ids))


class ForwardOnlyLocalUpdateTrace:
    """Sparse local credit trace that updates weights during forward steps only."""

    def __init__(
        self,
        *,
        capacity: int = 32,
        trace_decay: float = 0.75,
        learning_rate: float = 0.2,
        max_abs_weight: float = 1.0,
        min_abs_value: float = 0.001,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.trace_decay = max(0.0, min(1.0, float(trace_decay)))
        self.learning_rate = max(0.0, float(learning_rate))
        self.max_abs_weight = max(0.0, float(max_abs_weight))
        self.min_abs_value = max(0.0, float(min_abs_value))
        self._traces: Dict[Tuple[int, int], Dict[str, float]] = {}
        self._weights: Dict[Tuple[int, int], Dict[str, float]] = {}
        self._clock = 0

    def reset(self) -> None:
        self._traces.clear()
        self._weights.clear()
        self._clock = 0

    def update(
        self,
        *,
        pre_events: Sequence[Any],
        post_events: Sequence[Any],
        credit: float,
        retention_gate: float = 1.0,
    ) -> Dict[str, Any]:
        self._clock += 1
        pre_ids = _event_ids(pre_events)
        post_ids = _event_ids(post_events)
        local_credit = max(-1.0, min(1.0, float(credit)))
        retention = max(0.0, min(1.0, float(retention_gate)))

        decayed_trace_count = self._decay_traces()
        retained_weight_count = self._apply_retention(retention)
        trace_writes = 0
        weight_updates = 0
        for pre_id in pre_ids:
            for post_id in post_ids:
                key = (pre_id, post_id)
                trace = self._traces.get(key, {"eligibility": 0.0, "last_update": 0.0})
                trace["eligibility"] = min(1.0, float(trace["eligibility"]) + 1.0)
                trace["last_update"] = float(self._clock)
                self._traces[key] = trace
                trace_writes += 1

                weight = self._weights.get(key, {"weight": 0.0, "last_update": 0.0})
                delta = self.learning_rate * local_credit * float(trace["eligibility"])
                weight["weight"] = self._clip_weight(float(weight["weight"]) + delta)
                weight["last_update"] = float(self._clock)
                self._weights[key] = weight
                weight_updates += 1

        evicted_count = self._enforce_capacity()
        return {
            "pre_ids": pre_ids,
            "post_ids": post_ids,
            "credit": local_credit,
            "retention_gate": retention,
            "decayed_trace_count": decayed_trace_count,
            "retained_weight_count": retained_weight_count,
            "trace_writes": trace_writes,
            "weight_updates": weight_updates,
            "trace_units": len(self._traces),
            "weight_units": len(self._weights),
            "capacity": self.capacity,
            "state_budget_ok": self.state_budget_ok(),
            "evicted_count": evicted_count,
            "bptt_used": False,
            "clock": self._clock,
        }

    def read_weight(self, pre_id: int, post_id: int) -> float:
        return float(self._weights.get((int(pre_id), int(post_id)), {}).get("weight", 0.0))

    def state_budget_ok(self) -> bool:
        return len(self._traces) <= self.capacity and len(self._weights) <= self.capacity

    def snapshot(self) -> Dict[str, Any]:
        entries = []
        for key in sorted(set(self._traces) | set(self._weights)):
            trace = self._traces.get(key, {})
            weight = self._weights.get(key, {})
            entries.append(
                {
                    "pre_id": int(key[0]),
                    "post_id": int(key[1]),
                    "eligibility": float(trace.get("eligibility", 0.0)),
                    "weight": float(weight.get("weight", 0.0)),
                    "last_update": int(
                        max(
                            float(trace.get("last_update", 0.0)),
                            float(weight.get("last_update", 0.0)),
                        )
                    ),
                }
            )
        return {
            "schema": "sara-forward-only-local-update-trace-v1",
            "clock": self._clock,
            "capacity": self.capacity,
            "state_budget_ok": self.state_budget_ok(),
            "entries": entries,
        }

    def _clip_weight(self, value: float) -> float:
        return max(-self.max_abs_weight, min(self.max_abs_weight, float(value)))

    def _decay_traces(self) -> int:
        decayed = 0
        stale: List[Tuple[int, int]] = []
        for key, trace in self._traces.items():
            trace["eligibility"] = float(trace["eligibility"]) * self.trace_decay
            decayed += 1
            if abs(float(trace["eligibility"])) < self.min_abs_value:
                stale.append(key)
        for key in stale:
            self._traces.pop(key, None)
        return decayed

    def _apply_retention(self, retention_gate: float) -> int:
        retained = 0
        stale: List[Tuple[int, int]] = []
        for key, weight in self._weights.items():
            weight["weight"] = self._clip_weight(float(weight["weight"]) * retention_gate)
            retained += 1
            if abs(float(weight["weight"])) < self.min_abs_value:
                stale.append(key)
        for key in stale:
            self._weights.pop(key, None)
        return retained

    def _enforce_capacity(self) -> int:
        evicted = 0
        while len(self._traces) > self.capacity:
            remove_key = min(
                self._traces,
                key=lambda key: (
                    abs(float(self._traces[key]["eligibility"])),
                    float(self._traces[key]["last_update"]),
                ),
            )
            self._traces.pop(remove_key, None)
            evicted += 1
        while len(self._weights) > self.capacity:
            remove_key = min(
                self._weights,
                key=lambda key: (
                    abs(float(self._weights[key]["weight"])),
                    float(self._weights[key]["last_update"]),
                ),
            )
            self._weights.pop(remove_key, None)
            evicted += 1
        return evicted


def evaluate_forward_only_local_update_trace() -> Dict[str, Any]:
    trace = ForwardOnlyLocalUpdateTrace(capacity=4, learning_rate=0.25)
    positive_update = trace.update(pre_events=[1], post_events=[10], credit=1.0)
    reinforced_weight = trace.read_weight(1, 10)
    repeated_update = trace.update(pre_events=[1], post_events=[10], credit=1.0)
    damped_update = trace.update(
        pre_events=[1],
        post_events=[10],
        credit=-0.5,
        retention_gate=0.95,
    )
    budget_update = trace.update(pre_events=[2, 3, 4], post_events=[20, 30], credit=0.25)
    snapshot = trace.snapshot()
    weights = [abs(float(entry["weight"])) for entry in snapshot["entries"]]

    stability_ok = bool(
        reinforced_weight > 0.0
        and trace.read_weight(1, 10) <= trace.max_abs_weight
        and all(weight <= trace.max_abs_weight for weight in weights)
        and positive_update["bptt_used"] is False
        and repeated_update["bptt_used"] is False
        and damped_update["bptt_used"] is False
        and budget_update["state_budget_ok"]
        and snapshot["state_budget_ok"]
    )

    return {
        "observed_only": True,
        "metrics": {
            "forward_only_local_update_stability": 1.0 if stability_ok else 0.0,
            "forward_only_state_budget_integrity": (
                1.0 if bool(snapshot["state_budget_ok"]) else 0.0
            ),
        },
        "traces": {
            "positive_update": positive_update,
            "repeated_update": repeated_update,
            "damped_update": damped_update,
            "budget_update": budget_update,
            "snapshot": snapshot,
        },
    }
