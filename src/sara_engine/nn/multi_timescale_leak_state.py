# Directory Path: src/sara_engine/nn/multi_timescale_leak_state.py
# English Title: Multi-Timescale Leak State
# Purpose/Content: Bounded sparse membrane-state toy primitive for Linear RNN + SNN fusion probes.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple


def leak_event_ids(events: Sequence[Any]) -> List[int]:
    ids: List[int] = []
    for event in events:
        if hasattr(event, "spike_id"):
            ids.append(int(event.spike_id))
        elif isinstance(event, Mapping) and "spike_id" in event:
            ids.append(int(event["spike_id"]))
        else:
            ids.append(int(event))
    return sorted(set(ids))


DEFAULT_LEAK_GROUPS: Dict[str, Dict[str, float]] = {
    "short": {"leak_rate": 0.65, "threshold": 0.35, "input_gain": 1.0},
    "mid": {"leak_rate": 0.25, "threshold": 0.35, "input_gain": 1.0},
    "long": {"leak_rate": 0.05, "threshold": 0.35, "input_gain": 1.0},
}


class MultiTimescaleLeakState:
    """Sparse bounded membrane state with short, mid, and long leak groups."""

    def __init__(
        self,
        *,
        groups: Mapping[str, Mapping[str, float]] | None = None,
        max_state_units: int = 96,
        min_abs_value: float = 0.001,
    ) -> None:
        self.groups: Dict[str, Dict[str, float]] = {
            str(name): {
                "leak_rate": max(0.0, min(1.0, float(config.get("leak_rate", 0.0)))),
                "threshold": max(0.0, float(config.get("threshold", 0.0))),
                "input_gain": float(config.get("input_gain", 1.0)),
            }
            for name, config in (groups or DEFAULT_LEAK_GROUPS).items()
        }
        if not self.groups:
            raise ValueError("groups must not be empty")
        self.max_state_units = max(1, int(max_state_units))
        self.min_abs_value = max(0.0, float(min_abs_value))
        self._state: Dict[str, Dict[int, Dict[str, float]]] = {
            group_name: {} for group_name in self.groups
        }
        self._clock = 0

    def reset(self) -> None:
        for group_state in self._state.values():
            group_state.clear()
        self._clock = 0

    def update(
        self,
        input_events: Sequence[Any] = (),
        recurrent_events: Sequence[Any] = (),
        *,
        input_strength: float = 1.0,
        recurrent_strength: float = 0.5,
    ) -> Dict[str, Any]:
        self._clock += 1
        input_ids = leak_event_ids(input_events)
        recurrent_ids = leak_event_ids(recurrent_events)

        decayed_units = self._apply_leak()
        written_units = 0
        for group_name, config in self.groups.items():
            group_state = self._state[group_name]
            gain = float(config["input_gain"])
            for event_id in input_ids:
                written_units += self._add_value(
                    group_state,
                    event_id,
                    float(input_strength) * gain,
                )
            for event_id in recurrent_ids:
                written_units += self._add_value(
                    group_state,
                    event_id,
                    float(recurrent_strength) * gain,
                )

        evicted_units = self._enforce_state_budget()
        active_events = self.active_events()
        state_units = self.state_units()
        return {
            "input_ids": input_ids,
            "recurrent_ids": recurrent_ids,
            "decayed_units": decayed_units,
            "written_units": written_units,
            "evicted_units": evicted_units,
            "active_events": active_events,
            "state_units": state_units,
            "max_state_units": self.max_state_units,
            "state_budget_ok": state_units <= self.max_state_units,
            "clock": self._clock,
        }

    def step(self, count: int = 1) -> Dict[str, Any]:
        trace: Dict[str, Any] = {}
        for _ in range(max(1, int(count))):
            trace = self.update()
        return trace

    def read_event(self, event_id: int) -> Dict[str, float]:
        normalized_id = int(event_id)
        return {
            group_name: float(group_state.get(normalized_id, {}).get("value", 0.0))
            for group_name, group_state in self._state.items()
        }

    def active_events(self) -> Dict[str, List[int]]:
        active: Dict[str, List[int]] = {}
        for group_name, group_state in self._state.items():
            threshold = float(self.groups[group_name]["threshold"])
            active[group_name] = sorted(
                int(event_id)
                for event_id, entry in group_state.items()
                if abs(float(entry["value"])) >= threshold
            )
        return active

    def state_units(self) -> int:
        return sum(len(group_state) for group_state in self._state.values())

    def snapshot(self) -> Dict[str, Any]:
        groups: Dict[str, Any] = {}
        for group_name, group_state in sorted(self._state.items()):
            groups[group_name] = {
                "leak_rate": float(self.groups[group_name]["leak_rate"]),
                "threshold": float(self.groups[group_name]["threshold"]),
                "entries": [
                    {
                        "event_id": int(event_id),
                        "value": float(entry["value"]),
                        "last_update": int(entry["last_update"]),
                    }
                    for event_id, entry in sorted(group_state.items())
                ],
            }
        return {
            "schema": "sara-multi-timescale-leak-state-v1",
            "clock": int(self._clock),
            "state_units": self.state_units(),
            "max_state_units": self.max_state_units,
            "state_budget_ok": self.state_units() <= self.max_state_units,
            "groups": groups,
        }

    def _apply_leak(self) -> int:
        decayed_units = 0
        for group_name, group_state in self._state.items():
            leak_rate = float(self.groups[group_name]["leak_rate"])
            retention = max(0.0, 1.0 - leak_rate)
            stale_ids: List[int] = []
            for event_id, entry in group_state.items():
                entry["value"] = float(entry["value"]) * retention
                decayed_units += 1
                if abs(float(entry["value"])) < self.min_abs_value:
                    stale_ids.append(int(event_id))
            for event_id in stale_ids:
                group_state.pop(event_id, None)
        return decayed_units

    def _add_value(
        self,
        group_state: Dict[int, Dict[str, float]],
        event_id: int,
        value: float,
    ) -> int:
        normalized_id = int(event_id)
        current = group_state.get(normalized_id, {"value": 0.0, "last_update": 0.0})
        current["value"] = float(current["value"]) + float(value)
        current["last_update"] = float(self._clock)
        group_state[normalized_id] = current
        return 1

    def _enforce_state_budget(self) -> int:
        evicted = 0
        while self.state_units() > self.max_state_units:
            candidates: List[Tuple[float, float, str, int]] = []
            for group_name, group_state in self._state.items():
                for event_id, entry in group_state.items():
                    candidates.append(
                        (
                            abs(float(entry["value"])),
                            float(entry["last_update"]),
                            group_name,
                            int(event_id),
                        )
                    )
            if not candidates:
                break
            _, _, group_name, event_id = min(candidates)
            self._state[group_name].pop(event_id, None)
            evicted += 1
        return evicted


def evaluate_multi_timescale_leak_state() -> Dict[str, Any]:
    state = MultiTimescaleLeakState(max_state_units=12)
    first_trace = state.update(input_events=[101])
    state.step(count=3)
    values_after_decay = state.read_event(101)
    active_after_decay = state.active_events()
    budget_trace = state.update(input_events=[201, 202, 203, 204, 205])
    snapshot = state.snapshot()

    retention_ok = bool(
        values_after_decay["long"] > values_after_decay["mid"] > values_after_decay["short"]
    )
    active_long_ok = 1.0 if 101 in active_after_decay.get("long", []) else 0.0
    budget_ok = 1.0 if snapshot["state_budget_ok"] and budget_trace["state_budget_ok"] else 0.0

    return {
        "observed_only": True,
        "metrics": {
            "multi_timescale_leak_retention": 1.0 if retention_ok else 0.0,
            "multi_timescale_long_state_activity": active_long_ok,
            "timescale_state_budget_integrity": budget_ok,
        },
        "traces": {
            "first_update": first_trace,
            "values_after_decay": values_after_decay,
            "budget_update": budget_trace,
            "snapshot": snapshot,
        },
    }
