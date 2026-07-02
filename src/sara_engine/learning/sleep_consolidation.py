# Directory Path: src/sara_engine/learning/sleep_consolidation.py
# English Title: Sleep Consolidation Probe
# Purpose/Content: Evaluates bounded offline replay effects on retention, noise resilience, and energy cost.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class SleepConsolidationConfig:
    event_budget: float = 2.0
    min_retention: float = 0.70
    max_noise: float = 0.30
    min_health: float = 0.70
    min_branch_count: int = 2


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def evaluate_sleep_consolidation(
    replay_events: Iterable[Dict[str, Any]],
    config: SleepConsolidationConfig | None = None,
) -> Dict[str, Any]:
    """Evaluate offline replay without gradients, dense scans, or GPU assumptions."""

    cfg = config or SleepConsolidationConfig()
    traces: List[Dict[str, Any]] = []
    total_event_cost = 0.0
    for index, event in enumerate(replay_events):
        baseline_retention = _clamp01(float(event.get("baseline_retention", 0.0) or 0.0))
        post_retention = _clamp01(float(event.get("post_retention", baseline_retention) or 0.0))
        baseline_noise = _clamp01(float(event.get("baseline_noise", 1.0) or 0.0))
        post_noise = _clamp01(float(event.get("post_noise", baseline_noise) or 0.0))
        health_before = _clamp01(float(event.get("health_before", baseline_retention) or 0.0))
        health_after = _clamp01(float(event.get("health_after", post_retention) or 0.0))
        bundle_affinity = _clamp01(float(event.get("multimodal_bundle_affinity", 0.0) or 0.0))
        event_cost = max(0.0, float(event.get("event_cost", 0.0) or 0.0))
        branch_count = int(event.get("latent_branch_count", 1) or 1)
        selected_branch = str(event.get("selected_branch", ""))
        total_event_cost += event_cost
        traces.append(
            {
                "index": index,
                "memory_id": str(event.get("memory_id", event.get("id", f"memory-{index}"))),
                "baseline_retention": baseline_retention,
                "post_retention": post_retention,
                "retention_delta": post_retention - baseline_retention,
                "baseline_noise": baseline_noise,
                "post_noise": post_noise,
                "noise_delta": post_noise - baseline_noise,
                "health_before": health_before,
                "health_after": health_after,
                "health_delta": health_after - health_before,
                "multimodal_bundle_affinity": bundle_affinity,
                "event_cost": event_cost,
                "latent_branch_count": branch_count,
                "selected_branch": selected_branch,
                "retention_ok": post_retention >= max(cfg.min_retention, baseline_retention),
                "noise_ok": post_noise <= min(cfg.max_noise, baseline_noise),
                "health_ok": health_after >= max(cfg.min_health, health_before),
                "branch_ok": branch_count >= cfg.min_branch_count and bool(selected_branch),
            }
        )

    retention_ok = bool(traces) and all(bool(trace["retention_ok"]) for trace in traces)
    noise_ok = bool(traces) and all(bool(trace["noise_ok"]) for trace in traces)
    health_ok = bool(traces) and all(bool(trace["health_ok"]) for trace in traces)
    branch_ok = bool(traces) and any(bool(trace["branch_ok"]) for trace in traces)
    bundle_ok = bool(traces) and any(float(trace.get("multimodal_bundle_affinity", 0.0) or 0.0) > 0.0 for trace in traces)
    budget_ok = total_event_cost <= cfg.event_budget
    metrics = {
        "sleep_consolidation_retention_observed": 1.0 if retention_ok else 0.0,
        "latent_replay_noise_resilience_observed": 1.0 if noise_ok else 0.0,
        "sleep_consolidation_memory_health_observed": 1.0 if health_ok else 0.0,
        "latent_replay_counterfactual_branch_observed": 1.0 if branch_ok else 0.0,
        "multimodal_bundle_sleep_observed": 1.0 if bundle_ok else 0.0,
        "sleep_consolidation_energy_budget_observed": 1.0 if budget_ok else 0.0,
    }
    return {
        "observed_only": True,
        "config": {
            "event_budget": cfg.event_budget,
            "min_retention": cfg.min_retention,
            "max_noise": cfg.max_noise,
            "min_health": cfg.min_health,
            "min_branch_count": cfg.min_branch_count,
        },
        "traces": traces,
        "total_event_cost": float(total_event_cost),
        "event_budget_ok": bool(budget_ok),
        "metrics": metrics,
    }
