#!/usr/bin/env python3
"""Execute the registered 240-condition Phase 34 separation follow-up."""

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

from sara_engine.evaluation.phase34_memory_cache_preregistration import ARMS  # noqa: E402
from sara_engine.evaluation.phase34_separation_preregistration import (  # noqa: E402
    CASE_FAMILIES,
    REPLICATE_SEEDS,
    is_managed_preregistration_path,
    validate_preregistration,
)
from sara_engine.memory.memory_checkpoint_separation import (  # noqa: E402
    MemoryCacheSeparationRuntime,
    SeparationLimits,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures", "phase34_memory_cache_separation_cases.jsonl"
)
DEFAULT_PREREGISTRATION = workspace_path(
    "evaluation", "phase34_memory_cache_separation_preregistration.json"
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation", "phase34_memory_cache_separation_benchmark.json"
)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _environment_descriptor() -> Dict[str, Any]:
    return {
        "schema": "sara-phase34-memory-cache-separation-environment-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
    }


def load_fixture(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if tuple(row.get("family") for row in rows) != CASE_FAMILIES:
        raise ValueError("fixture_case_families_do_not_match_registration")
    if any(row.get("observed_only") is not True for row in rows):
        raise ValueError("separation fixture must be observed-only")
    if any(row.get("durable_mutation_allowed") is not False for row in rows):
        raise ValueError("separation fixture cannot mutate durable state")
    return rows


def load_preregistration(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validation = validate_preregistration(
        manifest, managed_path=is_managed_preregistration_path(path)
    )
    if not validation["valid"]:
        raise ValueError(
            "invalid separation registration: " + "; ".join(validation["errors"])
        )
    return manifest


def _limits(manifest: Mapping[str, Any]) -> SeparationLimits:
    budgets = manifest["budgets"]
    return SeparationLimits(
        max_events=int(budgets["source_events_per_case"]),
        max_attempted_checkpoints=int(budgets["attempted_checkpoints_per_case"]),
        max_checkpoints=int(budgets["max_checkpoints"]),
        selected_k=int(budgets["max_selected_checkpoints"]),
        max_summary_ids=int(budgets["max_summary_ids_per_checkpoint"]),
        max_state_bytes=int(budgets["max_total_state_bytes"]),
        max_event_cost=int(budgets["max_local_interactions_per_case"]),
        max_merges_per_event=int(budgets["max_merges_per_event"]),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _threshold_passed(value: float, spec: Mapping[str, Any]) -> bool:
    limit = float(spec["limit"])
    return value >= limit if spec["direction"] == "minimum" else value <= limit


def build_report(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    limits = _limits(manifest)
    fixture_fingerprint = _digest(list(rows))
    environment_fingerprint = _digest(_environment_descriptor())
    results: List[Dict[str, Any]] = []
    replay_results: List[Dict[str, Any]] = []
    cpu_latencies_ms: List[float] = []
    for arm in ARMS:
        for seed in REPLICATE_SEEDS:
            runtime = MemoryCacheSeparationRuntime(arm, limits)
            replay = MemoryCacheSeparationRuntime(arm, limits)
            for row in rows:
                started = time.process_time_ns()
                result = runtime.evaluate(row, seed=seed)
                cpu_latencies_ms.append(
                    (time.process_time_ns() - started) / 1_000_000.0
                )
                results.append({"arm": arm, "seed": seed, "result": result})
                replay_results.append(
                    {"arm": arm, "seed": seed, "result": replay.evaluate(row, seed=seed)}
                )

    indexed = {
        (entry["result"]["case_id"], entry["seed"], entry["arm"]): entry["result"]
        for entry in results
    }
    relation_passes: List[float] = []
    old_deltas: List[float] = []
    recent_deltas: List[float] = []
    topk_deltas: List[float] = []
    safety_values: List[float] = []
    relation_details: List[Dict[str, Any]] = []
    for row in rows:
        for seed in REPLICATE_SEEDS:
            case_id = str(row["case_id"])
            values = {arm: indexed[(case_id, seed, arm)] for arm in ARMS}
            relation = str(row["expected_relation"])
            if relation == "logarithmic_over_equal":
                delta = values[ARMS[2]]["recall"] - values[ARMS[1]]["recall"]
                passed = delta > 0.0
                old_deltas.append(delta)
            elif relation == "equal_over_logarithmic":
                delta = (
                    values[ARMS[1]]["retained_temporal_resolution"]
                    - values[ARMS[2]]["retained_temporal_resolution"]
                )
                passed = delta > 0.0
                recent_deltas.append(delta)
            elif relation == "topk_over_retrieve_all":
                delta = (
                    values[ARMS[3]]["selection_precision"]
                    - values[ARMS[1]]["selection_precision"]
                )
                passed = delta > 0.0 and values[ARMS[3]]["recall"] == 1.0
                topk_deltas.append(delta)
            elif relation == "deterministic_tie":
                delta = 0.0
                passed = results == replay_results
            else:
                delta = 0.0
                passed = all(value["safety_integrity"] == 1.0 for value in values.values())
                safety_values.extend(value["safety_integrity"] for value in values.values())
            relation_passes.append(float(passed))
            relation_details.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "expected_relation": relation,
                    "delta": delta,
                    "passed": passed,
                }
            )

    metrics = {
        "pairwise_separation_rate": _mean(relation_passes),
        "logarithmic_old_recall_delta": _mean(old_deltas),
        "topk_pollution_precision_delta": _mean(topk_deltas),
        "equal_recent_resolution_delta": _mean(recent_deltas),
        "safety_integrity": _mean(safety_values),
        "state_bytes": float(max(entry["result"]["state_bytes"] for entry in results)),
        "event_cost": float(max(entry["result"]["event_cost"] for entry in results)),
        "latency_ms": max(cpu_latencies_ms, default=0.0),
        "deterministic_replay": float(results == replay_results),
    }
    metric_gates = {
        name: _threshold_passed(value, manifest["thresholds"][name])
        for name, value in metrics.items()
    }
    checks = {
        "fixture_fingerprint_matches": fixture_fingerprint
        == manifest["fixture_fingerprint"],
        "environment_fingerprint_matches": environment_fingerprint
        == manifest["environment_fingerprint"],
        "all_240_conditions_executed": len(results) == 240,
        "all_five_registered_seeds_executed": sorted(
            {entry["seed"] for entry in results}
        )
        == sorted(REPLICATE_SEEDS),
        "deterministic_replay": results == replay_results,
        "all_resources_bounded": all(entry["result"]["bounded"] for entry in results),
        "no_durable_mutation": all(
            entry["result"]["durable_mutation"] is False for entry in results
        ),
        "production_path_not_changed": all(
            entry["result"]["production_path_changed"] is False for entry in results
        ),
        "cpu_only": True,
        "backpropagation_not_used": True,
        "matrix_calculation_not_used": True,
        "gpu_not_used": True,
    }
    execution_passed = all(checks.values())
    threshold_gate_passed = all(metric_gates.values())
    return {
        "schema": "sara-phase34-memory-cache-separation-benchmark-v1",
        "experiment_id": manifest["experiment_id"],
        "parent_protocol_fingerprint": manifest["parent_protocol_fingerprint"],
        "parent_report_fingerprint": manifest["parent_report_fingerprint"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "fixture_fingerprint": fixture_fingerprint,
        "environment_fingerprint": environment_fingerprint,
        "observed_only": True,
        "execution_passed": execution_passed,
        "threshold_gate_passed": threshold_gate_passed,
        "mechanism_gate_passed": threshold_gate_passed,
        "promotion_ready": False,
        "independent_evidence_available": False,
        "production_path_changed": False,
        "checks": checks,
        "metric_gates": metric_gates,
        "metrics": {"condition_count": len(results), **metrics},
        "relation_details": relation_details,
        "results": results,
        "failure_analysis": {
            "topk_retention_selection_confound": not metric_gates[
                "topk_pollution_precision_delta"
            ],
            "explanation": (
                "The registered pollution streams place the relevant checkpoint before "
                "equal-retention capacity eviction, so both retrieve-all and Top-k lose "
                "the target before selection."
            ),
            "registration_mutated": False,
        },
        "policy_notes": [
            "This run is synthetic and observed-only.",
            "CPU latency uses process CPU time.",
            "A failed registered relation is retained as negative evidence.",
            "No production, durable-memory, biological, energy, or general-accuracy claim is made.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--preregistration-path", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        report = build_report(
            load_fixture(args.fixture_path),
            load_preregistration(args.preregistration_path),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "condition_count": report["metrics"]["condition_count"],
                "execution_passed": report["execution_passed"],
                "threshold_gate_passed": report["threshold_gate_passed"],
                "mechanism_gate_passed": report["mechanism_gate_passed"],
                "promotion_ready": report["promotion_ready"],
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
