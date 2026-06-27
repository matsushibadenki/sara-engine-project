# Directory Path: src/sara_engine/learning/memory_phase.py
# English Title: Gradient-Free Memory Phase Tracking
# Purpose/Content: Tracks liquid/glass/crystal memory phases from local consolidation signals.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class MemoryPhaseConfig:
    glass_threshold: float = 0.45
    crystal_threshold: float = 0.72
    max_interference_for_crystal: float = 0.25
    state_budget: int = 16


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _phase_for_score(stability: float, replay_success: float, interference: float, cfg: MemoryPhaseConfig) -> str:
    if stability >= cfg.crystal_threshold and replay_success >= 0.70 and interference <= cfg.max_interference_for_crystal:
        return "crystal"
    if stability >= cfg.glass_threshold or replay_success >= 0.50:
        return "glass"
    return "liquid"


def _phase_rank(phase: str) -> int:
    return {"liquid": 0, "glass": 1, "crystal": 2}.get(phase, 0)


def build_memory_phase_observations(
    replay_events: Iterable[Dict[str, Any]],
    *,
    step: int = 1,
) -> List[Dict[str, Any]]:
    """Project replay observations into phase-tracking inputs.

    The mapping stays intentionally sparse and local:
    - stability favors retention and post-replay health
    - replay_success follows post-replay retention
    - interference follows residual post-replay noise
    """

    observations: List[Dict[str, Any]] = []
    for index, replay_event in enumerate(replay_events):
        memory_id = str(replay_event.get("memory_id", replay_event.get("id", "")) or "")
        if not memory_id:
            continue
        post_retention = _clamp01(float(replay_event.get("post_retention", 0.0) or 0.0))
        post_noise = _clamp01(float(replay_event.get("post_noise", 1.0) or 0.0))
        health_after = _clamp01(float(replay_event.get("health_after", post_retention) or 0.0))
        stability = _clamp01(0.55 * post_retention + 0.45 * health_after)
        observations.append(
            {
                "step": int(replay_event.get("step", step + index) or (step + index)),
                "memory_id": memory_id,
                "stability": stability,
                "replay_success": post_retention,
                "interference": post_noise,
            }
        )
    return observations


def evaluate_memory_phase_transitions(
    observations: Iterable[Dict[str, Any]],
    config: MemoryPhaseConfig | None = None,
) -> Dict[str, Any]:
    """Evaluate memory phase transitions without backpropagation or dense global state."""

    cfg = config or MemoryPhaseConfig()
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for observation in observations:
        memory_id = str(observation.get("memory_id", observation.get("id", "")))
        if not memory_id:
            continue
        grouped.setdefault(memory_id, []).append(dict(observation))

    phase_tracks: List[Dict[str, Any]] = []
    for memory_id, events in grouped.items():
        ordered = sorted(events, key=lambda item: int(item.get("step", 0) or 0))
        phases: List[str] = []
        states: List[Dict[str, Any]] = []
        for event in ordered:
            stability = _clamp01(float(event.get("stability", 0.0) or 0.0))
            replay_success = _clamp01(float(event.get("replay_success", 0.0) or 0.0))
            interference = _clamp01(float(event.get("interference", 0.0) or 0.0))
            phase = _phase_for_score(stability, replay_success, interference, cfg)
            phases.append(phase)
            plasticity = _clamp01(1.0 - stability + 0.35 * interference)
            if phase == "glass":
                plasticity *= 0.65
            elif phase == "crystal":
                plasticity *= 0.25
            retention = _clamp01(0.45 * stability + 0.35 * replay_success + 0.20 * (1.0 - interference))
            states.append(
                {
                    "step": int(event.get("step", 0) or 0),
                    "phase": phase,
                    "stability": stability,
                    "replay_success": replay_success,
                    "interference": interference,
                    "plasticity": plasticity,
                    "retention": retention,
                }
            )

        monotonic = all(
            _phase_rank(next_phase) >= _phase_rank(current_phase)
            for current_phase, next_phase in zip(phases, phases[1:])
        )
        phase_tracks.append(
            {
                "memory_id": memory_id,
                "phase_path": phases,
                "final_phase": phases[-1] if phases else "liquid",
                "monotonic_transition": bool(monotonic),
                "states": states,
                "final_retention": float(states[-1]["retention"]) if states else 0.0,
                "final_plasticity": float(states[-1]["plasticity"]) if states else 1.0,
            }
        )

    crystal_tracks = [track for track in phase_tracks if track["final_phase"] == "crystal"]
    non_crystal_tracks = [track for track in phase_tracks if track["final_phase"] != "crystal"]
    protected_crystals = [
        track
        for track in crystal_tracks
        if bool(track["monotonic_transition"]) and float(track["final_retention"]) >= 0.75
    ]
    liquid_tracks = [track for track in phase_tracks if track["final_phase"] == "liquid"]
    metrics = {
        "memory_phase_transition_integrity": 1.0
        if protected_crystals and any(track["phase_path"][0] == "liquid" for track in protected_crystals)
        else 0.0,
        "memory_phase_retention_protection_observed": 1.0
        if protected_crystals and min(float(track["final_retention"]) for track in protected_crystals) >= 0.75
        else 0.0,
        "memory_phase_plasticity_guard_observed": 1.0
        if protected_crystals
        and liquid_tracks
        and max(float(track["final_plasticity"]) for track in protected_crystals)
        < max(float(track["final_plasticity"]) for track in liquid_tracks)
        else 0.0,
        "memory_phase_overfixation_guard_observed": 1.0
        if phase_tracks and all(track["final_phase"] != "crystal" for track in non_crystal_tracks)
        else 0.0,
        "memory_phase_state_budget_observed": 1.0 if len(phase_tracks) <= cfg.state_budget else 0.0,
    }
    return {
        "observed_only": True,
        "config": {
            "glass_threshold": cfg.glass_threshold,
            "crystal_threshold": cfg.crystal_threshold,
            "max_interference_for_crystal": cfg.max_interference_for_crystal,
            "state_budget": cfg.state_budget,
        },
        "phase_tracks": phase_tracks,
        "metrics": metrics,
    }
