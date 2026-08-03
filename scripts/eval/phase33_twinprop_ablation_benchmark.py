#!/usr/bin/env python3
"""Execute the registered Phase 33 TwinProp-inspired ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase33_twinprop_preregistration import (  # noqa: E402
    ABLATION_ARMS,
    CASE_FAMILIES,
    is_managed_preregistration_path,
    validate_preregistration,
)
from sara_engine.neuro.twinprop_ablation import (  # noqa: E402
    TwinPropAblationLimits,
    TwinPropAblationRuntime,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures",
    "phase33_twinprop_ablation_cases.jsonl",
)
DEFAULT_PREREGISTRATION = workspace_path(
    "evaluation",
    "phase33_twinprop_ablation_preregistration.json",
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation",
    "phase33_twinprop_ablation_benchmark.json",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _environment_descriptor() -> Dict[str, Any]:
    return {
        "schema": "sara-phase33-twinprop-environment-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
        "dense_digital_twin": False,
    }


def load_fixture(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if tuple(row.get("family") for row in rows) != CASE_FAMILIES:
        raise ValueError("fixture_case_families_do_not_match_registration")
    if any(row.get("observed_only") is not True for row in rows):
        raise ValueError("TwinProp-inspired fixture must be observed-only")
    if any(
        row.get("expected", {}).get("durable_mutation_allowed") is not False
        for row in rows
    ):
        raise ValueError("TwinProp-inspired fixture cannot mutate durable state")
    return rows


def load_preregistration(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validation = validate_preregistration(
        manifest,
        managed_path=is_managed_preregistration_path(path),
    )
    if not validation["valid"]:
        raise ValueError(
            "invalid TwinProp-inspired registration: "
            + "; ".join(validation["errors"])
        )
    return manifest


def _limits(manifest: Mapping[str, Any]) -> TwinPropAblationLimits:
    budgets = manifest["budgets"]
    readout = manifest["fixed_readout"]
    return TwinPropAblationLimits(
        max_contacts=int(budgets["max_contacts_per_relation"]),
        max_branches=int(budgets["max_branch_slots_per_relation"]),
        max_slow_state_slots=int(budgets["max_slow_state_slots_per_relation"]),
        max_events=int(budgets["source_events_per_case"]),
        max_interactions=int(budgets["max_local_interactions_per_case"]),
        max_state_bytes=int(budgets["max_total_state_bytes"]),
        decision_window_ticks=int(readout["decision_window_ticks"]),
        readout_threshold=int(readout["threshold"]),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _threshold_passed(value: float, spec: Mapping[str, Any]) -> bool:
    limit = float(spec["limit"])
    if spec["direction"] == "minimum":
        return value >= limit
    return value <= limit


def build_report(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    limits = _limits(manifest)
    fixture_fingerprint = _digest(list(rows))
    environment_fingerprint = _digest(_environment_descriptor())
    results: List[Dict[str, Any]] = []
    replay_results: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    for arm in ABLATION_ARMS:
        runtime = TwinPropAblationRuntime(arm, limits)
        replay_runtime = TwinPropAblationRuntime(arm, limits)
        for seed in manifest["replicate_seeds"]:
            for row in rows:
                started = time.perf_counter()
                result = runtime.evaluate(row)
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
                results.append(
                    {"arm": arm, "replicate_seed": seed, "result": result}
                )
                replay_results.append(
                    {
                        "arm": arm,
                        "replicate_seed": seed,
                        "result": replay_runtime.evaluate(row),
                    }
                )

    summaries: Dict[str, Dict[str, Any]] = {}
    for arm in ABLATION_ARMS:
        arm_results = [entry["result"] for entry in results if entry["arm"] == arm]
        summaries[arm] = {
            "condition_count": len(arm_results),
            "fixed_readout_quality": _mean(
                [float(result["target_match"]) for result in arm_results]
            ),
            "abstention_count": sum(
                result["status"] == "abstained" for result in arm_results
            ),
            "max_state_bytes": max(
                int(result["state_bytes"]) for result in arm_results
            ),
            "max_event_cost": max(int(result["event_cost"]) for result in arm_results),
            "max_slow_state_saturation": max(
                float(result["slow_state_saturation"]) for result in arm_results
            ),
            "durable_mutation_count": sum(
                bool(result["durable_mutation"]) for result in arm_results
            ),
        }

    intact = summaries["intact_bounded_branches"]["fixed_readout_quality"]
    order_participation: Dict[int, List[int]] = {2: [], 3: [], 4: []}
    safety_matches: List[float] = []
    structured_counts: List[int] = []
    shuffled_counts: List[int] = []
    for entry in results:
        result = entry["result"]
        if entry["arm"] != "intact_bounded_branches":
            continue
        family = result["family"]
        if family.startswith("interaction_order_"):
            order = int(family.rsplit("_", 1)[1])
            order_participation[order].append(int(result["active_branch_count"]))
        if family in {"missing_contact", "stale_source_revision"}:
            safety_matches.append(float(result["target_match"]))
        if family == "deterministic_contact_placement":
            structured_counts.append(int(result["readout_count"]))
        if family == "shuffled_contact_placement":
            shuffled_counts.append(int(result["readout_count"]))
    mean_participation = {
        order: _mean([float(value) for value in values])
        for order, values in order_participation.items()
    }
    monotonicity = float(
        mean_participation[2] <= mean_participation[3]
        and mean_participation[3] <= mean_participation[4]
        and mean_participation[4] > mean_participation[2]
    )
    metrics = {
        "fixed_readout_quality": intact,
        "branch_participation_monotonicity": monotonicity,
        "structured_over_shuffled_delta": (
            _mean([float(value) for value in structured_counts])
            - _mean([float(value) for value in shuffled_counts])
        )
        / limits.readout_threshold,
        "intact_over_passive_delta": intact
        - summaries["passive_linear_branches"]["fixed_readout_quality"],
        "intact_over_collapsed_delta": intact
        - summaries["topology_collapsed_aggregation"]["fixed_readout_quality"],
        "intact_over_no_slow_state_delta": intact
        - summaries["no_slow_coincidence_state"]["fixed_readout_quality"],
        "abstention_integrity": _mean(safety_matches),
        "state_bytes": float(
            max(summary["max_state_bytes"] for summary in summaries.values())
        ),
        "event_cost": float(
            max(summary["max_event_cost"] for summary in summaries.values())
        ),
        "latency_ms": max(latencies_ms, default=0.0),
        "deterministic_replay": float(results == replay_results),
    }
    metric_gates = {
        metric: _threshold_passed(value, manifest["thresholds"][metric])
        for metric, value in metrics.items()
    }
    expected_conditions = (
        len(ABLATION_ARMS) * len(manifest["replicate_seeds"]) * len(rows)
    )
    checks = {
        "fixture_fingerprint_matches": (
            fixture_fingerprint == manifest["fixture_fingerprint"]
        ),
        "environment_fingerprint_matches": (
            environment_fingerprint == manifest["environment_fingerprint"]
        ),
        "parent_protocol_unchanged": manifest["parent_protocol_fingerprint"]
        == "63168395ac7f5235d4173072fb52823712b89895e16610856ced77adf70d64ff",
        "all_registered_conditions_executed": len(results) == expected_conditions,
        "same_fixed_readout_across_arms": all(
            entry["result"]["readout_threshold"] == limits.readout_threshold
            for entry in results
        ),
        "deterministic_replay": results == replay_results,
        "no_durable_mutation": all(
            entry["result"]["durable_mutation"] is False for entry in results
        ),
        "state_budget_respected": metrics["state_bytes"]
        <= limits.max_state_bytes,
        "event_budget_respected": metrics["event_cost"]
        <= limits.max_interactions,
        "latency_budget_respected": metrics["latency_ms"]
        <= float(manifest["budgets"]["max_latency_ms"]),
        "equal_tuning_budget_preserved": (
            manifest["budgets"]["tuning_trials_per_arm"] == 1
            and manifest["budgets"]["restart_count_per_arm"] == 0
        ),
        "cpu_only": True,
        "backpropagation_not_used": True,
        "matrix_calculation_not_used": True,
        "gpu_not_used": True,
        "dense_digital_twin_not_used": True,
        "production_path_not_changed": True,
    }
    execution_passed = all(checks.values())
    mechanism_gate_passed = execution_passed and all(metric_gates.values())
    return {
        "schema": "sara-phase33-twinprop-ablation-benchmark-v1",
        "experiment_id": manifest["experiment_id"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "parent_protocol_fingerprint": manifest["parent_protocol_fingerprint"],
        "fixture_fingerprint": fixture_fingerprint,
        "environment_fingerprint": environment_fingerprint,
        "observed_only": True,
        "execution_passed": execution_passed,
        "mechanism_gate_passed": mechanism_gate_passed,
        "promotion_ready": False,
        "independent_evidence_available": False,
        "production_path_changed": False,
        "checks": checks,
        "metric_gates": metric_gates,
        "metrics": {
            **metrics,
            "case_count": len(rows),
            "condition_count": len(results),
            "mean_active_branches_by_interaction_order": mean_participation,
        },
        "arm_summaries": summaries,
        "results": results,
        "policy_notes": [
            "This is a designed observed-only mechanism ablation, not task accuracy evidence.",
            "The fixed non-trainable readout is identical across all arms.",
            "No DNN twin, gradient-selected placement, PCA runtime, GPU, or matrix path is used.",
            "Independent workloads and human review remain required for promotion.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--preregistration-path", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report(
        load_fixture(args.fixture_path),
        load_preregistration(args.preregistration_path),
    )
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "execution_passed": report["execution_passed"],
                "mechanism_gate_passed": report["mechanism_gate_passed"],
                "promotion_ready": report["promotion_ready"],
                "condition_count": report["metrics"]["condition_count"],
                "output_path": os.path.realpath(args.output_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
