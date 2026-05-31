# Directory Path: src/sara_engine/learning/astro_structural_gate.py
# English Title: Astro Structural Plasticity Gate
# Purpose/Content: Observes prediction-error gated structural plasticity unlock/lock policy.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .astro_modulator import AstroReplayModulator


@dataclass(frozen=True)
class AstroStructuralGateConfig:
    unlock_error_threshold: float = 0.65
    lock_error_threshold: float = 0.25
    min_stability_for_lock: float = 0.55
    max_policy_events: int = 8


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def evaluate_astro_structural_gate(
    replay_steps: Iterable[Dict[str, Any]],
    config: AstroStructuralGateConfig | None = None,
) -> Dict[str, Any]:
    """Evaluate astrocyte-inspired structural unlock/lock policy without gradients."""

    cfg = config or AstroStructuralGateConfig()
    modulator = AstroReplayModulator(stress_decay=0.82, support_recovery=0.18, stress_gain=0.42)
    structural_unlocked = False
    policy_trace: List[Dict[str, Any]] = []

    for index, raw_step in enumerate(replay_steps):
        step = dict(raw_step)
        prediction_error = _clamp01(float(step.get("prediction_error", 0.0) or 0.0))
        recovery = _clamp01(float(step.get("replay_recovery", step.get("world_model_recovery", 0.0)) or 0.0))
        modulator.update(interference_ratio=prediction_error, replay_recovery_signal=recovery)
        astro_state = modulator.snapshot()

        action = "hold"
        if prediction_error >= cfg.unlock_error_threshold:
            structural_unlocked = True
            action = "unlock_structural_plasticity"
        elif structural_unlocked and (
            prediction_error <= cfg.lock_error_threshold
            and float(astro_state.get("stability_level", 0.0)) >= cfg.min_stability_for_lock
        ):
            structural_unlocked = False
            action = "lock_to_bounded_stdp"
        elif not structural_unlocked:
            action = "bounded_stdp_only"

        policy_trace.append(
            {
                "index": index,
                "prediction_error": prediction_error,
                "replay_recovery": recovery,
                "action": action,
                "structural_unlocked": bool(structural_unlocked),
                "astro_state": astro_state,
                "world_model_event": str(step.get("world_model_event", step.get("event", ""))),
            }
        )

    actions = [str(item.get("action", "")) for item in policy_trace]
    unlock_indices = [index for index, action in enumerate(actions) if action == "unlock_structural_plasticity"]
    lock_indices = [index for index, action in enumerate(actions) if action == "lock_to_bounded_stdp"]
    ordered_unlock_lock = bool(unlock_indices and lock_indices and min(lock_indices) > min(unlock_indices))
    final_locked = bool(policy_trace and not bool(policy_trace[-1].get("structural_unlocked", True)))
    metrics = {
        "astro_structural_unlock_observed": 1.0 if unlock_indices else 0.0,
        "astro_structural_lock_observed": 1.0 if ordered_unlock_lock and final_locked else 0.0,
        "astro_bounded_stdp_fallback_observed": 1.0 if "bounded_stdp_only" in actions or final_locked else 0.0,
        "world_model_replay_policy_trace_observed": 1.0
        if policy_trace and all(str(item.get("world_model_event", "")) for item in policy_trace)
        else 0.0,
        "astro_policy_state_budget_observed": 1.0 if len(policy_trace) <= cfg.max_policy_events else 0.0,
    }
    return {
        "observed_only": True,
        "config": {
            "unlock_error_threshold": cfg.unlock_error_threshold,
            "lock_error_threshold": cfg.lock_error_threshold,
            "min_stability_for_lock": cfg.min_stability_for_lock,
            "max_policy_events": cfg.max_policy_events,
        },
        "policy_trace": policy_trace,
        "final_structural_unlocked": bool(policy_trace[-1]["structural_unlocked"]) if policy_trace else False,
        "metrics": metrics,
    }
