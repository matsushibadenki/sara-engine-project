#!/usr/bin/env python3
"""Compare generic and relation-class-aware bounded structural plasticity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.learning.structural_plasticity import BoundedStructuralPlasticityController  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_REPORT_PATH = workspace_path("evaluation", "risa_structural_plasticity_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path(
    "evaluation", "risa_structural_plasticity_benchmark_summary.txt"
)
PREDICTS_ROUTE = (11, 12)
OBSERVES_ROUTE = (21, 22)


def _frozen_fixture() -> Dict[str, Any]:
    return {
        "schema": "sara-risa-structural-plasticity-fixture-v1",
        "replay_budget": 2,
        "max_rewrites_per_event": 4,
        "routes": {
            "predicts": PREDICTS_ROUTE,
            "observes": OBSERVES_ROUTE,
        },
        "global_signals": {
            "prediction_error": 0.78,
            "novelty": 0.35,
            "reward": 0.62,
            "contradiction": 0.72,
            "metabolic_headroom": 0.86,
            "source_backed": True,
        },
        "relation_contradiction_pressure": {
            "predicts": 0.10,
            "observes": 0.88,
        },
    }


def _controller() -> BoundedStructuralPlasticityController:
    controller = BoundedStructuralPlasticityController(
        max_total_links=4,
        max_fan_in=2,
        max_fan_out=2,
        max_rewrites_per_event=4,
        prune_grace_steps=1,
        contradiction_growth_block=0.55,
        contradiction_prune_threshold=2,
        min_stable_verified_support=2,
        min_stable_prediction_gain=0.65,
    )
    for route in (PREDICTS_ROUTE, OBSERVES_ROUTE):
        controller.register_route(
            route,
            weight=0.7,
            route_state="provisional",
            responsibility=0.8,
            longevity=0.5,
            prediction_gain_support=0.5,
            contradiction_count=1,
            support_count=1,
            verified_support_count=1,
            created_step=-2,
            last_active_step=0,
        )
    return controller


def _event_inputs(fixture: Mapping[str, Any]) -> Dict[str, Any]:
    support = {
        route: {
            "prediction_gain_support": 0.30,
            "replay_support": 0.72,
            "verified": True,
        }
        for route in (PREDICTS_ROUTE, OBSERVES_ROUTE)
    }
    return {
        "active_routes": {PREDICTS_ROUTE: 0.9, OBSERVES_ROUTE: 0.35},
        "signals": dict(fixture["global_signals"]),
        "event_memory_support": support,
    }


def _run_variant(
    fixture: Mapping[str, Any],
    *,
    relation_class_aware: bool,
) -> Dict[str, Any]:
    controller = _controller()
    inputs = _event_inputs(fixture)
    pressure = None
    if relation_class_aware:
        pressure = {
            PREDICTS_ROUTE: float(fixture["relation_contradiction_pressure"]["predicts"]),
            OBSERVES_ROUTE: float(fixture["relation_contradiction_pressure"]["observes"]),
        }
    traces = []
    for replay_index in range(int(fixture["replay_budget"])):
        if replay_index:
            inputs = {**inputs, "active_routes": {PREDICTS_ROUTE: 0.0, OBSERVES_ROUTE: 0.0}}
        result = controller.apply_event(
            **inputs,
            route_contradiction_pressure=pressure,
        )
        traces.append(result.to_dict())
    predicts_state = controller.routes.get(PREDICTS_ROUTE)
    observes_state = controller.routes.get(OBSERVES_ROUTE)
    return {
        "mode": "relation_class_aware" if relation_class_aware else "generic",
        "traces": traces,
        "final_routes": controller.snapshot(),
        "predictive_route_retained": predicts_state is not None,
        "predictive_route_stable": bool(
            predicts_state is not None and predicts_state.route_state == "stable"
        ),
        "contradictory_route_pruned": observes_state is None,
        "total_event_cost": sum(int(trace["event_cost"]) for trace in traces),
        "total_rewrites": sum(int(trace["trace"]["actions_taken"]) for trace in traces),
        "max_state_budget_units": max(
            (int(trace["state_budget_units"]) for trace in traces),
            default=0,
        ),
    }


def build_report() -> Dict[str, Any]:
    fixture = _frozen_fixture()
    fixture_bytes = json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode("utf-8")
    generic = _run_variant(fixture, relation_class_aware=False)
    relation_aware = _run_variant(fixture, relation_class_aware=True)
    retention_improved = bool(
        relation_aware["predictive_route_stable"]
        and not generic["predictive_route_retained"]
    )
    contradiction_recovery_maintained = bool(
        relation_aware["contradictory_route_pruned"]
        and generic["contradictory_route_pruned"]
    )
    maintenance_cost_equal = (
        int(relation_aware["total_event_cost"]) == int(generic["total_event_cost"])
    )
    rewrites_bounded = (
        int(relation_aware["total_rewrites"])
        <= int(fixture["replay_budget"]) * int(fixture["max_rewrites_per_event"])
    )
    state_bounded = int(relation_aware["max_state_budget_units"]) <= 4
    passed = all(
        (
            retention_improved,
            contradiction_recovery_maintained,
            maintenance_cost_equal,
            rewrites_bounded,
            state_bounded,
        )
    )
    return {
        "schema": "sara-risa-structural-plasticity-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "frozen_fixture": True,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "replay_budget": int(fixture["replay_budget"]),
        "metrics": {
            "predictive_route_retention_improved": float(retention_improved),
            "contradiction_recovery_maintained": float(contradiction_recovery_maintained),
            "maintenance_cost_equal": float(maintenance_cost_equal),
            "rewrites_bounded": float(rewrites_bounded),
            "state_bounded": float(state_bounded),
            "generic_event_cost": int(generic["total_event_cost"]),
            "relation_class_aware_event_cost": int(relation_aware["total_event_cost"]),
            "generic_rewrites": int(generic["total_rewrites"]),
            "relation_class_aware_rewrites": int(relation_aware["total_rewrites"]),
        },
        "variants": {
            "generic": generic,
            "relation_class_aware": relation_aware,
        },
        "policy_notes": [
            "Both variants use the identical frozen replay sequence, source-backed signals, and rewrite budget.",
            "The benchmark is observed-only and does not change production structural state.",
            "A pass is fixture evidence, not promotion evidence for independent long-horizon workloads.",
        ],
    }


def summarize(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    return "\n".join(
        (
            f"RISA structural plasticity benchmark: {'PASS' if report['passed'] else 'FAIL'}",
            f"Frozen fixture: {report['frozen_fixture']}",
            f"Replay budget: {report['replay_budget']}",
            f"Predictive retention improved: {metrics['predictive_route_retention_improved']}",
            f"Contradiction recovery maintained: {metrics['contradiction_recovery_maintained']}",
            f"Maintenance cost equal: {metrics['maintenance_cost_equal']}",
        )
    ) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report()
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    summary_path = ensure_parent_directory(args.summary_path)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summarize(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
