# Directory Path: src/sara_engine/nn/phase_synchronized_binding_trace.py
# English Title: Phase Synchronized Binding Trace
# Purpose/Content: Bounded sparse coincidence trace for phase-based event binding probes.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _event_field(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(name, default)
    return getattr(event, name, default)


def _normalize_event(event: Any, index: int, phase_buckets: int) -> Dict[str, Any]:
    event_id = int(_event_field(event, "event_id", _event_field(event, "spike_id", index)))
    raw_phase = int(_event_field(event, "phase", event_id))
    step = int(_event_field(event, "step", index))
    return {
        "event_id": event_id,
        "phase": raw_phase,
        "phase_bucket": raw_phase % max(1, int(phase_buckets)),
        "step": step,
        "label": str(_event_field(event, "label", "")),
    }


class PhaseSynchronizedBindingTrace:
    """Sparse phase-bucket coincidence checker for distant event binding."""

    def __init__(
        self,
        *,
        phase_buckets: int = 8,
        min_temporal_distance: int = 16,
        max_bindings: int = 32,
    ) -> None:
        self.phase_buckets = max(1, int(phase_buckets))
        self.min_temporal_distance = max(0, int(min_temporal_distance))
        self.max_bindings = max(1, int(max_bindings))

    def build(
        self,
        *,
        anchor_events: Sequence[Any],
        candidate_events: Sequence[Any],
    ) -> Dict[str, Any]:
        anchors = [
            _normalize_event(event, index, self.phase_buckets)
            for index, event in enumerate(anchor_events)
        ]
        candidates = [
            _normalize_event(event, index, self.phase_buckets)
            for index, event in enumerate(candidate_events)
        ]

        bindings: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for anchor in anchors:
            for candidate in candidates:
                temporal_distance = abs(int(candidate["step"]) - int(anchor["step"]))
                same_phase = anchor["phase_bucket"] == candidate["phase_bucket"]
                distant = temporal_distance >= self.min_temporal_distance
                pair = {
                    "anchor_id": int(anchor["event_id"]),
                    "candidate_id": int(candidate["event_id"]),
                    "anchor_bucket": int(anchor["phase_bucket"]),
                    "candidate_bucket": int(candidate["phase_bucket"]),
                    "temporal_distance": temporal_distance,
                    "same_phase_bucket": bool(same_phase),
                    "distant_enough": bool(distant),
                }
                if same_phase and distant and len(bindings) < self.max_bindings:
                    bindings.append(pair)
                else:
                    rejected.append(pair)

        return {
            "schema": "sara-phase-synchronized-binding-trace-v1",
            "observed_only": True,
            "phase_buckets": self.phase_buckets,
            "min_temporal_distance": self.min_temporal_distance,
            "max_bindings": self.max_bindings,
            "anchors": anchors,
            "candidates": candidates,
            "bindings": bindings,
            "rejected": rejected,
            "binding_count": len(bindings),
            "state_budget_ok": len(bindings) <= self.max_bindings,
        }


def evaluate_phase_synchronized_binding_trace() -> Dict[str, Any]:
    trace_builder = PhaseSynchronizedBindingTrace(
        phase_buckets=8,
        min_temporal_distance=32,
        max_bindings=4,
    )
    trace = trace_builder.build(
        anchor_events=[
            {"event_id": 101, "phase": 3, "step": 0, "label": "subject"},
        ],
        candidate_events=[
            {"event_id": 301, "phase": 11, "step": 96, "label": "matching_action"},
            {"event_id": 302, "phase": 4, "step": 104, "label": "distractor"},
            {"event_id": 303, "phase": 3, "step": 8, "label": "near_noise"},
        ],
    )
    bound_pairs: List[Tuple[int, int]] = [
        (int(binding["anchor_id"]), int(binding["candidate_id"]))
        for binding in trace["bindings"]
    ]
    rejected_pairs: List[Tuple[int, int]] = [
        (int(pair["anchor_id"]), int(pair["candidate_id"]))
        for pair in trace["rejected"]
    ]
    expected_pair_ok = (101, 301) in bound_pairs
    phase_guard_ok = (101, 302) in rejected_pairs
    distance_guard_ok = (101, 303) in rejected_pairs

    return {
        "observed_only": True,
        "metrics": {
            "phase_binding_coincidence_integrity": (
                1.0 if expected_pair_ok and phase_guard_ok and distance_guard_ok else 0.0
            ),
            "phase_binding_state_budget_integrity": (
                1.0 if bool(trace["state_budget_ok"]) else 0.0
            ),
        },
        "traces": {
            "phase_synchronized_binding_trace": trace,
            "bound_pairs": bound_pairs,
            "rejected_pairs": rejected_pairs,
        },
    }
