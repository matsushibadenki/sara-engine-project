# Directory Path: src/sara_engine/learning/metabolic_budget.py
# English Title: Structural Metabolic Budget
# Purpose/Content: Evaluates bounded structural plasticity operations with resource pressure and pruning reasons.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class MetabolicBudgetConfig:
    max_synapses: int = 8
    event_budget: float = 4.0
    plasticity_reserve: float = 1.0
    high_pressure_threshold: float = 0.80
    min_growth_importance: float = 0.45


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def evaluate_structural_metabolic_budget(
    operations: Iterable[Dict[str, Any]],
    config: MetabolicBudgetConfig | None = None,
) -> Dict[str, Any]:
    """Evaluate structural plasticity operations under bounded metabolic resources."""

    cfg = config or MetabolicBudgetConfig()
    synapse_count = 0
    event_cost_used = 0.0
    reserve_used = 0.0
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    pruning_trace: List[Dict[str, Any]] = []

    for index, raw_operation in enumerate(operations):
        operation = dict(raw_operation)
        kind = str(operation.get("kind", operation.get("operation", "grow")))
        synapse_delta = int(operation.get("synapse_delta", 1 if kind in {"grow", "rewire"} else -1) or 0)
        event_cost = max(0.0, float(operation.get("event_cost", 0.0) or 0.0))
        reserve_cost = max(0.0, float(operation.get("reserve_cost", event_cost * 0.25) or 0.0))
        importance = _clamp01(float(operation.get("importance", 0.0) or 0.0))
        projected_synapses = max(0, synapse_count + synapse_delta)
        projected_event_cost = event_cost_used + event_cost
        projected_reserve = reserve_used + reserve_cost
        pressure = max(
            projected_synapses / max(int(cfg.max_synapses), 1),
            projected_event_cost / max(float(cfg.event_budget), 1e-9),
            projected_reserve / max(float(cfg.plasticity_reserve), 1e-9),
        )

        reason = "accepted"
        accept = True
        if kind == "prune":
            reason = str(operation.get("reason", "low_importance_prune"))
            pruning_trace.append(
                {
                    "index": index,
                    "reason": reason,
                    "importance": importance,
                    "resource_pressure": _clamp01(pressure),
                }
            )
        elif projected_synapses > cfg.max_synapses:
            accept = False
            reason = "synapse_budget_limit"
        elif projected_event_cost > cfg.event_budget:
            accept = False
            reason = "event_budget_limit"
        elif projected_reserve > cfg.plasticity_reserve:
            accept = False
            reason = "plasticity_reserve_limit"
        elif pressure >= cfg.high_pressure_threshold and importance < cfg.min_growth_importance:
            accept = False
            reason = "low_importance_under_resource_pressure"

        record = {
            "index": index,
            "kind": kind,
            "accepted": bool(accept),
            "reason": reason,
            "synapse_delta": synapse_delta,
            "event_cost": event_cost,
            "reserve_cost": reserve_cost,
            "importance": importance,
            "resource_pressure": _clamp01(pressure),
        }
        if accept:
            synapse_count = projected_synapses
            event_cost_used = projected_event_cost
            reserve_used = projected_reserve
            accepted.append(record)
        else:
            rejected.append(record)

    final_pressure = max(
        synapse_count / max(int(cfg.max_synapses), 1),
        event_cost_used / max(float(cfg.event_budget), 1e-9),
        reserve_used / max(float(cfg.plasticity_reserve), 1e-9),
    )
    has_growth_rejection = any(
        str(item.get("reason")) in {"synapse_budget_limit", "event_budget_limit", "plasticity_reserve_limit", "low_importance_under_resource_pressure"}
        for item in rejected
    )
    metrics = {
        "metabolic_budget_integrity": 1.0
        if synapse_count <= cfg.max_synapses and event_cost_used <= cfg.event_budget
        else 0.0,
        "plasticity_reserve_integrity": 1.0 if reserve_used <= cfg.plasticity_reserve else 0.0,
        "structural_growth_bounded_observed": 1.0 if has_growth_rejection else 0.0,
        "pruning_reason_trace_observed": 1.0 if pruning_trace and all(item["reason"] for item in pruning_trace) else 0.0,
        "resource_pressure_observed": 1.0 if final_pressure > 0.0 else 0.0,
    }
    return {
        "observed_only": True,
        "config": {
            "max_synapses": cfg.max_synapses,
            "event_budget": cfg.event_budget,
            "plasticity_reserve": cfg.plasticity_reserve,
            "high_pressure_threshold": cfg.high_pressure_threshold,
            "min_growth_importance": cfg.min_growth_importance,
        },
        "accepted_operations": accepted,
        "rejected_operations": rejected,
        "pruning_trace": pruning_trace,
        "resource_pressure": _clamp01(final_pressure),
        "synapse_count": int(synapse_count),
        "event_cost_used": float(event_cost_used),
        "plasticity_reserve_used": float(reserve_used),
        "metrics": metrics,
    }
