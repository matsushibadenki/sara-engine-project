"""Bounded default-off runtime controls for Phase 30 temporal interactions."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from typing import Any, Deque, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .phase30_preregistration import ARMS


RUNTIME_SCHEMA = "sara-phase30-temporal-control-result-v1"
INVALIDATING_KINDS = frozenset({"context_revision", "contradiction", "expiry"})


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass
class EdgeTemporalState:
    edge_id: str
    recent: Deque[Tuple[int, int, int, str]]
    last_timestamp_us: Optional[int] = None
    last_phase_bucket: Optional[int] = None
    excitation: float = 0.0
    fatigue: float = 0.0
    reuse_count: int = 0
    last_order: int = -1
    provenance_reference: str = ""


@dataclass(frozen=True)
class CachedInteraction:
    edge_id: str
    value: float
    built_order: int
    expiry_order: int
    provenance_reference: str


@dataclass
class TemporalControlRuntime:
    arm: str
    max_active_edges: int = 64
    max_recent_events_per_edge: int = 16
    max_cached_interactions: int = 32
    reuse_threshold: int = 2
    max_event_cost: int = 4096
    timestamp_resolution_us: int = 1000
    states: "OrderedDict[str, EdgeTemporalState]" = field(default_factory=OrderedDict)
    cache: "OrderedDict[str, CachedInteraction]" = field(default_factory=OrderedDict)
    invalidation_trace: List[Dict[str, Any]] = field(default_factory=list)
    event_cost: int = 0
    cache_hits: int = 0
    cache_builds: int = 0
    direct_computes: int = 0
    phase_discontinuities: int = 0
    nonmonotonic_events: int = 0
    duplicate_events: int = 0
    unknown_context_events: int = 0
    _last_global_timestamp: Optional[int] = None
    _last_source_event_id: Optional[str] = None
    _fixed_sum: float = 0.0
    _history_sum: float = 0.0
    _history_count: int = 0
    _temporal_sum: float = 0.0

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError("unknown_phase30_arm")
        for value in (
            self.max_active_edges,
            self.max_recent_events_per_edge,
            self.max_cached_interactions,
            self.reuse_threshold,
            self.max_event_cost,
            self.timestamp_resolution_us,
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("runtime_limits_must_be_positive_integers")

    def _edge_state(self, edge_id: str) -> EdgeTemporalState:
        state = self.states.get(edge_id)
        if state is not None:
            self.states.move_to_end(edge_id)
            return state
        if len(self.states) >= self.max_active_edges:
            evicted_edge, _ = self.states.popitem(last=False)
            self._invalidate(evicted_edge, "active_edge_eviction", -1)
        state = EdgeTemporalState(edge_id=edge_id, recent=deque(maxlen=self.max_recent_events_per_edge))
        self.states[edge_id] = state
        return state

    def _invalidate(self, edge_id: str, reason: str, order: int) -> None:
        removed = self.cache.pop(edge_id, None)
        self.event_cost += 1
        self.invalidation_trace.append(
            {
                "edge_id": edge_id,
                "reason": reason,
                "order": order,
                "cache_entry_removed": removed is not None,
            }
        )

    def _invalidate_all(self, reason: str, order: int) -> None:
        for edge_id in tuple(self.cache):
            self._invalidate(edge_id, reason, order)
        if not self.cache:
            self.invalidation_trace.append(
                {"edge_id": "*", "reason": reason, "order": order, "cache_entry_removed": False}
            )
            self.event_cost += 1

    def _temporal_value(self, state: EdgeTemporalState, event: Mapping[str, Any]) -> float:
        timestamp = int(event["timestamp_us"])
        phase = int(event["phase_bucket"])
        polarity = int(event["polarity"])
        if state.last_timestamp_us is None:
            interval_steps = 1.0
            phase_alignment = 1.0
        else:
            delta = max(0, timestamp - state.last_timestamp_us)
            interval_steps = max(1.0, delta / self.timestamp_resolution_us)
            expected_phase = (int(state.last_phase_bucket or 0) + 1) % 8
            phase_distance = min((phase - expected_phase) % 8, (expected_phase - phase) % 8)
            phase_alignment = 1.0 - (phase_distance / 4.0)
            if phase_distance >= 3:
                self.phase_discontinuities += 1
        decay = 1.0 / (1.0 + interval_steps / 8.0)
        state.excitation = _clip(state.excitation * decay + 0.20 * abs(polarity), 0.0, 1.0)
        state.fatigue = _clip(state.fatigue * 0.92 + 0.04 * state.reuse_count, 0.0, 1.0)
        value = polarity * decay * (0.5 + 0.5 * phase_alignment) * (0.5 + state.excitation) * (1.0 - 0.5 * state.fatigue)
        return _clip(value, -1.0, 1.0)

    def _cached_or_direct(self, state: EdgeTemporalState, event: Mapping[str, Any], value: float) -> float:
        order = int(event["order"])
        cached = self.cache.get(state.edge_id)
        if cached is not None and order <= cached.expiry_order:
            self.cache.move_to_end(state.edge_id)
            self.cache_hits += 1
            self.event_cost += 1
            return cached.value
        if cached is not None:
            self._invalidate(state.edge_id, "cache_expiry", order)
        self.direct_computes += 1
        self.event_cost += 2
        if state.reuse_count >= self.reuse_threshold:
            if len(self.cache) >= self.max_cached_interactions:
                evicted_edge, _ = self.cache.popitem(last=False)
                self.invalidation_trace.append(
                    {"edge_id": evicted_edge, "reason": "cache_capacity", "order": order, "cache_entry_removed": True}
                )
                self.event_cost += 1
            self.cache[state.edge_id] = CachedInteraction(
                edge_id=state.edge_id,
                value=value,
                built_order=order,
                expiry_order=order + self.max_recent_events_per_edge,
                provenance_reference=str(event["provenance_reference"]),
            )
            self.cache_builds += 1
            self.event_cost += 2
        return value

    def observe(self, event: Mapping[str, Any]) -> None:
        required = (
            "source_event_id",
            "edge_id",
            "timestamp_us",
            "order",
            "phase_bucket",
            "polarity",
            "kind",
            "provenance_reference",
        )
        if any(name not in event for name in required):
            raise ValueError("phase30_event_contract_incomplete")
        timestamp = int(event["timestamp_us"])
        order = int(event["order"])
        phase = int(event["phase_bucket"])
        polarity = int(event["polarity"])
        if phase < 0 or phase > 7 or polarity not in (-1, 0, 1):
            raise ValueError("phase30_event_scalar_out_of_range")
        if self._last_global_timestamp is not None and timestamp < self._last_global_timestamp:
            self.nonmonotonic_events += 1
        if str(event["kind"]) == "duplicate" or str(event["source_event_id"]) == self._last_source_event_id:
            self.duplicate_events += 1
        if str(event["kind"]) == "unknown_context":
            self.unknown_context_events += 1
        self._last_global_timestamp = timestamp
        self._last_source_event_id = str(event["source_event_id"])
        self.event_cost += 1
        if self.event_cost > self.max_event_cost:
            raise RuntimeError("phase30_event_cost_exceeded")

        edge_id = str(event["edge_id"])
        state = self._edge_state(edge_id)
        if str(event["kind"]) in INVALIDATING_KINDS:
            self._invalidate_all(str(event["kind"]), order)
        state.reuse_count += 1
        value = self._temporal_value(state, event)
        state.recent.append((timestamp, phase, polarity, str(event["provenance_reference"])))
        state.last_timestamp_us = timestamp
        state.last_phase_bucket = phase
        state.last_order = order
        state.provenance_reference = str(event["provenance_reference"])

        self._fixed_sum += polarity
        self._history_sum += polarity
        self._history_count += 1
        if self.arm == "fixed_sparse_snn":
            contribution = float(polarity)
        elif self.arm == "history_averaged_static_interaction":
            contribution = self._history_sum / self._history_count
        elif self.arm == "temporal_state_only":
            contribution = value
            self.direct_computes += 1
            self.event_cost += 2
        else:
            contribution = self._cached_or_direct(state, event, value)
        self._temporal_sum += contribution
        if self.event_cost > self.max_event_cost:
            raise RuntimeError("phase30_event_cost_exceeded")

    def finish(self, case_id: str) -> Dict[str, Any]:
        if self._history_count == 0:
            raise ValueError("phase30_empty_history")
        if self.arm == "fixed_sparse_snn":
            raw_score = self._fixed_sum / self._history_count
        elif self.arm == "history_averaged_static_interaction":
            raw_score = self._history_sum / self._history_count
        else:
            raw_score = self._temporal_sum / self._history_count
        uncertainty_flags = self.nonmonotonic_events + self.unknown_context_events
        if uncertainty_flags > 0 or abs(raw_score) < 0.05:
            decision = "abstain"
        else:
            decision = "positive" if raw_score > 0 else "negative"
        state_payload = {
            edge_id: {
                "recent": list(state.recent),
                "excitation": round(state.excitation, 12),
                "fatigue": round(state.fatigue, 12),
                "reuse_count": state.reuse_count,
                "last_order": state.last_order,
                "provenance_reference": state.provenance_reference,
            }
            for edge_id, state in self.states.items()
        }
        cache_payload = {
            edge_id: {
                "value": round(item.value, 12),
                "built_order": item.built_order,
                "expiry_order": item.expiry_order,
                "provenance_reference": item.provenance_reference,
            }
            for edge_id, item in self.cache.items()
        }
        result: Dict[str, Any] = {
            "schema": RUNTIME_SCHEMA,
            "case_id": case_id,
            "arm": self.arm,
            "decision": decision,
            "score": round(_clip(raw_score, -1.0, 1.0), 12),
            "event_count": self._history_count,
            "event_cost": self.event_cost,
            "active_edge_count": len(self.states),
            "cached_interaction_count": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_builds": self.cache_builds,
            "direct_computes": self.direct_computes,
            "phase_discontinuities": self.phase_discontinuities,
            "nonmonotonic_events": self.nonmonotonic_events,
            "duplicate_events": self.duplicate_events,
            "unknown_context_events": self.unknown_context_events,
            "state_bytes": len(json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")),
            "cache_bytes": len(json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")),
            "state": state_payload,
            "cache": cache_payload,
            "invalidation_trace": self.invalidation_trace,
            "production_mutation": False,
            "durable_knowledge_mutation": False,
        }
        result["replay_digest"] = _canonical_digest(result)
        return result


def run_control(case: Mapping[str, Any], arm: str) -> Dict[str, Any]:
    runtime = TemporalControlRuntime(arm=arm)
    for event in case.get("events", ()):
        runtime.observe(event)
    return runtime.finish(str(case["case_id"]))


__all__ = ["INVALIDATING_KINDS", "RUNTIME_SCHEMA", "TemporalControlRuntime", "run_control"]
