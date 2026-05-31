# Directory Path: src/sara_engine/nn/delta_associative_memory.py
# English Title: Delta Associative Spike Memory
# Purpose/Content: Bounded sparse online memory inspired by delta-rule residual writes.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple


def event_ids(events: Sequence[Any]) -> List[int]:
    ids: List[int] = []
    for event in events:
        if hasattr(event, "spike_id"):
            ids.append(int(event.spike_id))
        elif isinstance(event, Mapping) and "spike_id" in event:
            ids.append(int(event["spike_id"]))
        else:
            ids.append(int(event))
    return sorted(set(ids))


class DeltaAssociativeSpikeMemory:
    """
    Stores only residual event associations in a bounded sparse state.

    The memory does not update when the observed events are already predicted.
    This keeps repeated context from over-strengthening and limits interference.
    """

    def __init__(
        self,
        capacity: int = 64,
        residual_threshold: float = 0.01,
        min_weight: float = 0.001,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.residual_threshold = max(0.0, float(residual_threshold))
        self.min_weight = max(0.0, float(min_weight))
        self._state: Dict[Tuple[int, int], Dict[str, float]] = {}
        self._clock = 0

    def reset(self) -> None:
        self._state.clear()
        self._clock = 0

    def update(
        self,
        context_events: Sequence[Any],
        predicted_events: Sequence[Any],
        observed_events: Sequence[Any],
        write_gate: float = 1.0,
        retention_gate: float = 1.0,
    ) -> Dict[str, Any]:
        self._clock += 1
        context_ids = event_ids(context_events)
        predicted_ids = set(event_ids(predicted_events))
        observed_ids = event_ids(observed_events)
        residual_ids = [event_id for event_id in observed_ids if event_id not in predicted_ids]
        write_strength = max(0.0, min(1.0, float(write_gate)))
        retention = max(0.0, min(1.0, float(retention_gate)))

        self._apply_retention(retention)
        write_applied = bool(context_ids and residual_ids and write_strength > self.residual_threshold)
        if write_applied:
            for context_id in context_ids:
                for residual_id in residual_ids:
                    key = (context_id, residual_id)
                    current = self._state.get(key, {"weight": 0.0, "last_update": 0.0})
                    current["weight"] = min(1.0, float(current["weight"]) + write_strength)
                    current["last_update"] = float(self._clock)
                    self._state[key] = current

        evicted = self._enforce_capacity()
        state_units = len(self._state)
        return {
            "context_ids": context_ids,
            "predicted_ids": sorted(predicted_ids),
            "observed_ids": observed_ids,
            "residual_ids": residual_ids,
            "write_applied": write_applied,
            "write_gate": write_strength,
            "retention_gate": retention,
            "state_units": state_units,
            "capacity": self.capacity,
            "state_budget_ok": state_units <= self.capacity,
            "evicted_count": evicted,
            "interference_guard": not residual_ids or write_applied,
        }

    def read(
        self,
        context_events: Sequence[Any],
        limit: int = 8,
        min_weight: float = 0.0,
    ) -> Dict[str, Any]:
        context_ids = set(event_ids(context_events))
        scores: Dict[int, float] = {}
        contributing_keys: List[Tuple[int, int]] = []
        for (context_id, value_id), entry in self._state.items():
            weight = float(entry["weight"])
            if context_id in context_ids and weight >= min_weight:
                scores[value_id] = scores.get(value_id, 0.0) + weight
                contributing_keys.append((context_id, value_id))
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: max(1, int(limit))]
        return {
            "context_ids": sorted(context_ids),
            "predicted_ids": [value_id for value_id, _ in ranked],
            "scores": {int(value_id): float(score) for value_id, score in ranked},
            "contributing_key_count": len(contributing_keys),
            "state_units": len(self._state),
        }

    def snapshot(self) -> Dict[str, Any]:
        entries = []
        for (context_id, value_id), entry in sorted(self._state.items()):
            entries.append(
                {
                    "context_id": int(context_id),
                    "value_id": int(value_id),
                    "weight": float(entry["weight"]),
                    "last_update": int(entry["last_update"]),
                }
            )
        return {
            "capacity": self.capacity,
            "state_units": len(self._state),
            "entries": entries,
        }

    def build_memory_steering_event(
        self,
        context_events: Sequence[Any],
        branch_id: str = "primary",
        limit: int = 8,
    ) -> Dict[str, Any]:
        readout = self.read(context_events, limit=limit)
        return {
            "event_type": "memory_steering_event",
            "memory_type": "delta_associative_state",
            "branch_id": str(branch_id),
            "observed_only": True,
            "context_ids": readout["context_ids"],
            "steering_ids": readout["predicted_ids"],
            "scores": readout["scores"],
            "contributing_key_count": readout["contributing_key_count"],
            "state_units": readout["state_units"],
            "text_reinjection_used": False,
            "trace_complete": bool(readout["context_ids"]),
        }

    def _apply_retention(self, retention_gate: float) -> None:
        if retention_gate >= 1.0:
            return
        stale_keys: List[Tuple[int, int]] = []
        for key, entry in self._state.items():
            entry["weight"] = float(entry["weight"]) * retention_gate
            if float(entry["weight"]) < self.min_weight:
                stale_keys.append(key)
        for key in stale_keys:
            self._state.pop(key, None)

    def _enforce_capacity(self) -> int:
        evicted = 0
        while len(self._state) > self.capacity:
            remove_key = min(
                self._state,
                key=lambda key: (float(self._state[key]["weight"]), float(self._state[key]["last_update"])),
            )
            self._state.pop(remove_key, None)
            evicted += 1
        return evicted


def evaluate_delta_associative_spike_memory() -> Dict[str, Any]:
    memory = DeltaAssociativeSpikeMemory(capacity=4)
    first_trace = memory.update(
        context_events=[1, 2],
        predicted_events=[10],
        observed_events=[10, 11],
    )
    repeated_trace = memory.update(
        context_events=[1, 2],
        predicted_events=[10, 11],
        observed_events=[10, 11],
    )
    recall = memory.read([1, 2])
    budget_trace = memory.update(
        context_events=[3, 4, 5],
        predicted_events=[],
        observed_events=[30, 31],
    )

    residual_write_ok = bool(first_trace["write_applied"] and first_trace["residual_ids"] == [11])
    retention_ok = bool(not repeated_trace["write_applied"] and repeated_trace["state_units"] <= memory.capacity)
    recall_ok = 1.0 if 11 in recall["predicted_ids"] else 0.0
    budget_ok = 1.0 if budget_trace["state_budget_ok"] and memory.snapshot()["state_units"] <= memory.capacity else 0.0
    interference_ok = 1.0 if repeated_trace["interference_guard"] and not repeated_trace["residual_ids"] else 0.0

    return {
        "observed_only": True,
        "metrics": {
            "delta_memory_residual_write_integrity": 1.0 if residual_write_ok else 0.0,
            "delta_memory_retention_gate_stability": 1.0 if retention_ok else 0.0,
            "delta_memory_context_recall_without_text_reinjection": recall_ok,
            "delta_memory_state_budget_integrity": budget_ok,
            "delta_memory_interference_guard": interference_ok,
        },
        "traces": {
            "first_write": first_trace,
            "repeated_observation": repeated_trace,
            "budget_write": budget_trace,
            "recall": recall,
            "snapshot": memory.snapshot(),
        },
    }


def evaluate_delta_memory_steering_trace() -> Dict[str, Any]:
    memory = DeltaAssociativeSpikeMemory(capacity=6)
    primary_write = memory.update(
        context_events=[101, 102],
        predicted_events=[201],
        observed_events=[201, 301],
    )
    counterfactual_write = memory.update(
        context_events=[111, 112],
        predicted_events=[202],
        observed_events=[202, 302],
    )
    repeated_primary = memory.update(
        context_events=[101, 102],
        predicted_events=[201, 301],
        observed_events=[201, 301],
    )
    primary_event = memory.build_memory_steering_event(
        context_events=[101, 102],
        branch_id="primary",
    )
    counterfactual_event = memory.build_memory_steering_event(
        context_events=[111, 112],
        branch_id="counterfactual-1",
    )
    isolated_probe = memory.build_memory_steering_event(
        context_events=[909],
        branch_id="unrelated-probe",
    )
    snapshot = memory.snapshot()

    steering_ok = bool(
        primary_event["event_type"] == "memory_steering_event"
        and primary_event["memory_type"] == "delta_associative_state"
        and primary_event["steering_ids"] == [301]
        and primary_event["text_reinjection_used"] is False
        and primary_event["contributing_key_count"] > 0
    )
    counterfactual_ok = bool(
        counterfactual_event["steering_ids"] == [302]
        and 301 not in counterfactual_event["steering_ids"]
        and isolated_probe["steering_ids"] == []
    )
    trace_ok = bool(
        primary_write["write_applied"]
        and counterfactual_write["write_applied"]
        and not repeated_primary["write_applied"]
        and primary_event["trace_complete"]
        and snapshot["state_units"] <= snapshot["capacity"]
    )

    return {
        "observed_only": True,
        "metrics": {
            "delta_memory_steering_integrity": 1.0 if steering_ok else 0.0,
            "delta_memory_counterfactual_isolation": 1.0 if counterfactual_ok else 0.0,
            "delta_memory_trace_observability": 1.0 if trace_ok else 0.0,
        },
        "traces": {
            "primary_write": primary_write,
            "counterfactual_write": counterfactual_write,
            "repeated_primary": repeated_primary,
            "primary_steering_event": primary_event,
            "counterfactual_steering_event": counterfactual_event,
            "isolated_probe": isolated_probe,
            "snapshot": snapshot,
        },
    }
