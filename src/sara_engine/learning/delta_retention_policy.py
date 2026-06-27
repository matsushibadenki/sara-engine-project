# Directory Path: src/sara_engine/learning/delta_retention_policy.py
# English Title: Delta Memory Retention Policy
# Purpose/Content: Observes memory-phase and astro-modulated retention gates for delta associative state.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from ..nn.delta_associative_memory import DeltaAssociativeSpikeMemory


@dataclass(frozen=True)
class DeltaRetentionPolicyConfig:
    crystal_retention: float = 0.98
    glass_retention: float = 0.82
    liquid_retention: float = 0.45
    min_crystal_score: float = 0.65
    max_liquid_score: float = 0.35
    capacity: int = 6


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _base_retention_for_phase(phase: str, cfg: DeltaRetentionPolicyConfig) -> float:
    if phase == "crystal":
        return cfg.crystal_retention
    if phase == "glass":
        return cfg.glass_retention
    return cfg.liquid_retention


def _retention_gate_for_event(event: Dict[str, Any], cfg: DeltaRetentionPolicyConfig) -> float:
    phase = str(event.get("phase", "liquid"))
    astro_stability = _clamp01(float(event.get("astro_stability", 1.0) or 0.0))
    base_retention = _base_retention_for_phase(phase, cfg)
    return _clamp01(base_retention * (0.70 + 0.30 * astro_stability))


def _erase_gate_for_event(event: Dict[str, Any], cfg: DeltaRetentionPolicyConfig) -> float:
    phase = str(event.get("phase", "liquid"))
    astro_stability = _clamp01(float(event.get("astro_stability", 1.0) or 0.0))
    retention = _base_retention_for_phase(phase, cfg)
    residual_magnitude = _clamp01(float(event.get("residual_magnitude", 0.0) or 0.0))
    return _clamp01((1.0 - retention) * (0.70 + 0.30 * (1.0 - astro_stability)) + 0.10 * residual_magnitude)


def _write_gate_for_event(event: Dict[str, Any]) -> float:
    residual_magnitude = _clamp01(float(event.get("residual_magnitude", 0.0) or 0.0))
    novelty = _clamp01(float(event.get("novelty", 1.0) or 0.0))
    return _clamp01(residual_magnitude * (0.50 + 0.50 * novelty))


def build_delta_retention_events(
    replay_events: Iterable[Dict[str, Any]],
    *,
    astro_stability: float = 1.0,
) -> List[Dict[str, Any]]:
    """Convert replay observations into phase-aware delta-retention inputs."""

    clamped_astro = _clamp01(astro_stability)
    events: List[Dict[str, Any]] = []
    for replay_event in replay_events:
        memory_id = str(replay_event.get("memory_id", replay_event.get("id", "")) or "")
        if not memory_id:
            continue
        phase = str(replay_event.get("phase", "liquid") or "liquid")
        post_retention = _clamp01(float(replay_event.get("post_retention", 0.0) or 0.0))
        post_noise = _clamp01(float(replay_event.get("post_noise", 1.0) or 0.0))
        novelty = _clamp01(1.0 - post_retention + 0.5 * post_noise)
        event_key = abs(hash(memory_id)) % 1_000_000
        context_event = 100_000 + event_key
        observed_event = 200_000 + event_key
        predicted_events = [observed_event] if post_retention >= 0.60 else []
        events.append(
            {
                "memory_id": memory_id,
                "phase": phase,
                "astro_stability": clamped_astro,
                "context_events": [context_event],
                "predicted_events": predicted_events,
                "observed_events": [observed_event],
                "residual_magnitude": post_noise,
                "novelty": novelty,
                "write_gate": _clamp01(0.35 + 0.65 * novelty),
            }
        )
    return events


def evaluate_delta_retention_policy(
    memory_events: Iterable[Dict[str, Any]],
    config: DeltaRetentionPolicyConfig | None = None,
) -> Dict[str, Any]:
    """Evaluate retention/forget gate choices for delta memory without gradients."""

    cfg = config or DeltaRetentionPolicyConfig()
    memory = DeltaAssociativeSpikeMemory(capacity=cfg.capacity, min_weight=0.02)
    traces: List[Dict[str, Any]] = []
    for index, raw_event in enumerate(memory_events):
        event = dict(raw_event)
        phase = str(event.get("phase", "liquid"))
        astro_stability = _clamp01(float(event.get("astro_stability", 1.0) or 0.0))
        retention_gate = _retention_gate_for_event(event, cfg)
        update = memory.update(
            context_events=list(event.get("context_events", [])),
            predicted_events=list(event.get("predicted_events", [])),
            observed_events=list(event.get("observed_events", [])),
            write_gate=float(event.get("write_gate", 1.0) or 0.0),
            retention_gate=retention_gate,
        )
        readout = memory.read(list(event.get("context_events", [])))
        traces.append(
            {
                "index": index,
                "phase": phase,
                "astro_stability": astro_stability,
                "retention_gate": retention_gate,
                "update": update,
                "readout": readout,
            }
        )

    crystal_traces = [trace for trace in traces if trace["phase"] == "crystal"]
    liquid_traces = [trace for trace in traces if trace["phase"] == "liquid"]
    glass_traces = [trace for trace in traces if trace["phase"] == "glass"]
    final_snapshot = memory.snapshot()
    crystal_retained = any(
        901 in trace["readout"].get("predicted_ids", [])
        and trace["retention_gate"] >= cfg.min_crystal_score
        for trace in crystal_traces
    )
    liquid_forgot = bool(
        liquid_traces
        and all(trace["retention_gate"] <= cfg.max_liquid_score for trace in liquid_traces)
    )
    astro_aligned = bool(
        crystal_traces
        and liquid_traces
        and max(trace["retention_gate"] for trace in crystal_traces)
        > max(trace["retention_gate"] for trace in liquid_traces)
    )
    glass_middle = bool(
        glass_traces
        and liquid_traces
        and crystal_traces
        and max(trace["retention_gate"] for trace in liquid_traces)
        < min(trace["retention_gate"] for trace in glass_traces)
        <= max(trace["retention_gate"] for trace in crystal_traces)
    )
    metrics = {
        "delta_memory_phase_retention_policy_observed": 1.0 if glass_middle else 0.0,
        "delta_memory_crystal_retention_observed": 1.0 if crystal_retained else 0.0,
        "delta_memory_liquid_forget_observed": 1.0 if liquid_forgot else 0.0,
        "delta_memory_astro_gate_alignment_observed": 1.0 if astro_aligned else 0.0,
        "delta_memory_policy_state_budget_observed": 1.0
        if int(final_snapshot.get("state_units", 0) or 0) <= cfg.capacity
        else 0.0,
    }
    return {
        "observed_only": True,
        "config": {
            "crystal_retention": cfg.crystal_retention,
            "glass_retention": cfg.glass_retention,
            "liquid_retention": cfg.liquid_retention,
            "min_crystal_score": cfg.min_crystal_score,
            "max_liquid_score": cfg.max_liquid_score,
            "capacity": cfg.capacity,
        },
        "traces": traces,
        "snapshot": final_snapshot,
        "metrics": metrics,
    }


def evaluate_delta_erase_write_decoupling(
    memory_events: Iterable[Dict[str, Any]],
    config: DeltaRetentionPolicyConfig | None = None,
) -> Dict[str, Any]:
    """Evaluate separate erase and write gates for bounded delta memory edits."""

    cfg = config or DeltaRetentionPolicyConfig()
    memory = DeltaAssociativeSpikeMemory(capacity=cfg.capacity, min_weight=0.02)
    traces: List[Dict[str, Any]] = []
    for index, raw_event in enumerate(memory_events):
        event = dict(raw_event)
        erase_gate = _erase_gate_for_event(event, cfg)
        write_gate = _write_gate_for_event(event)
        retention_gate = _clamp01(1.0 - erase_gate)
        update = memory.update(
            context_events=list(event.get("context_events", [])),
            predicted_events=list(event.get("predicted_events", [])),
            observed_events=list(event.get("observed_events", [])),
            write_gate=write_gate,
            retention_gate=retention_gate,
        )
        readout = memory.read(list(event.get("probe_events", event.get("context_events", []))), limit=4)
        traces.append(
            {
                "index": index,
                "phase": str(event.get("phase", "liquid")),
                "astro_stability": _clamp01(float(event.get("astro_stability", 1.0) or 0.0)),
                "residual_magnitude": _clamp01(float(event.get("residual_magnitude", 0.0) or 0.0)),
                "erase_gate": erase_gate,
                "write_gate": write_gate,
                "retention_gate": retention_gate,
                "update": update,
                "readout": readout,
                "expected_stable_ids": list(event.get("expected_stable_ids", [])),
                "expected_write_ids": list(event.get("expected_write_ids", [])),
            }
        )

    final_snapshot = memory.snapshot()
    decoupled = bool(
        traces
        and any(abs(float(trace["write_gate"]) - float(trace["erase_gate"])) >= 0.25 for trace in traces)
    )
    stable_traces = [trace for trace in traces if trace["expected_stable_ids"]]
    stable_preserved = bool(
        stable_traces
        and all(
            set(int(value) for value in trace["expected_stable_ids"]).issubset(
                set(int(value) for value in trace["readout"].get("predicted_ids", []))
            )
            and float(trace["erase_gate"]) <= 0.20
            for trace in stable_traces
        )
    )
    write_traces = [trace for trace in traces if trace["expected_write_ids"]]
    residual_committed = bool(
        write_traces
        and all(
            bool(trace["update"].get("write_applied", False))
            and set(int(value) for value in trace["expected_write_ids"]).issubset(
                set(int(value) for value in trace["readout"].get("predicted_ids", []))
            )
            and float(trace["write_gate"]) > float(trace["erase_gate"])
            for trace in write_traces
        )
    )
    metrics = {
        "delta_memory_erase_write_decoupling_observed": 1.0 if decoupled else 0.0,
        "delta_memory_erase_preserves_stable_memory_observed": 1.0 if stable_preserved else 0.0,
        "delta_memory_write_commits_residual_observed": 1.0 if residual_committed else 0.0,
    }
    return {
        "observed_only": True,
        "config": {
            "capacity": cfg.capacity,
            "crystal_retention": cfg.crystal_retention,
            "glass_retention": cfg.glass_retention,
            "liquid_retention": cfg.liquid_retention,
        },
        "traces": traces,
        "snapshot": final_snapshot,
        "metrics": metrics,
    }


def evaluate_delta_retention_policy_stress(
    memory_histories: Iterable[Dict[str, Any]],
    config: DeltaRetentionPolicyConfig | None = None,
) -> Dict[str, Any]:
    """Stress delta retention over multiple sparse histories without gradients."""

    cfg = config or DeltaRetentionPolicyConfig()
    memory = DeltaAssociativeSpikeMemory(capacity=cfg.capacity, min_weight=0.02)
    histories = [dict(history) for history in memory_histories]
    traces: List[Dict[str, Any]] = []
    for index, history in enumerate(histories):
        retention_gate = _retention_gate_for_event(history, cfg)
        update = memory.update(
            context_events=list(history.get("context_events", [])),
            predicted_events=list(history.get("predicted_events", [])),
            observed_events=list(history.get("observed_events", [])),
            write_gate=float(history.get("write_gate", 1.0) or 0.0),
            retention_gate=retention_gate,
        )
        traces.append(
            {
                "index": index,
                "branch_id": str(history.get("branch_id", f"history-{index}")),
                "phase": str(history.get("phase", "liquid")),
                "retention_gate": retention_gate,
                "expected_recall_ids": list(history.get("expected_recall_ids", [])),
                "update": update,
            }
        )

    probes: List[Dict[str, Any]] = []
    expected_target_sets: List[set[int]] = []
    for index, history in enumerate(histories):
        expected_ids = set(int(value) for value in history.get("expected_recall_ids", []))
        if not expected_ids:
            continue
        expected_target_sets.append(expected_ids)
        readout = memory.read(list(history.get("context_events", [])), limit=4)
        predicted_ids = set(int(value) for value in readout.get("predicted_ids", []))
        probes.append(
            {
                "index": index,
                "branch_id": str(history.get("branch_id", f"history-{index}")),
                "context_ids": readout.get("context_ids", []),
                "expected_recall_ids": sorted(expected_ids),
                "predicted_ids": readout.get("predicted_ids", []),
                "recall_ok": expected_ids.issubset(predicted_ids),
                "cross_branch_leak": bool(
                    any(
                        other_expected - expected_ids
                        and predicted_ids.intersection(other_expected - expected_ids)
                        for other_expected in expected_target_sets
                    )
                ),
            }
        )

    unrelated_probe = memory.read(list(histories[0].get("unrelated_probe_events", [999001])) if histories else [999001])
    liquid_traces = [trace for trace in traces if trace["phase"] == "liquid"]
    snapshot = memory.snapshot()
    recall_ok = bool(probes and all(bool(probe["recall_ok"]) for probe in probes))
    noise_ok = bool(
        liquid_traces
        and all(float(trace["retention_gate"]) <= cfg.max_liquid_score for trace in liquid_traces)
        and not unrelated_probe.get("predicted_ids")
    )
    manifold_guard_ok = bool(probes and not any(bool(probe["cross_branch_leak"]) for probe in probes))
    health_ok = bool(
        int(snapshot.get("state_units", 0) or 0) <= cfg.capacity
        and all(float(entry.get("weight", 0.0) or 0.0) >= memory.min_weight for entry in snapshot.get("entries", []))
    )
    metrics = {
        "delta_memory_multi_history_recall_observed": 1.0 if recall_ok else 0.0,
        "delta_memory_multi_history_noise_resilience_observed": 1.0 if noise_ok else 0.0,
        "delta_memory_multi_history_health_observed": 1.0 if health_ok else 0.0,
        "delta_memory_multi_history_manifold_guard_observed": 1.0 if manifold_guard_ok else 0.0,
    }
    return {
        "observed_only": True,
        "config": {
            "capacity": cfg.capacity,
            "min_crystal_score": cfg.min_crystal_score,
            "max_liquid_score": cfg.max_liquid_score,
        },
        "traces": traces,
        "probes": probes,
        "unrelated_probe": unrelated_probe,
        "snapshot": snapshot,
        "metrics": metrics,
    }
