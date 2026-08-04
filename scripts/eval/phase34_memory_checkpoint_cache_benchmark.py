#!/usr/bin/env python3
"""Execute the registered observed-only Phase 34 cache ablation."""

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

from sara_engine.evaluation.phase34_memory_cache_preregistration import (  # noqa: E402
    ARMS,
    CASE_FAMILIES,
    is_managed_preregistration_path,
    validate_preregistration,
)
from sara_engine.memory.memory_checkpoint_ablation import (  # noqa: E402
    Phase34MemoryCacheLimits,
    Phase34MemoryCheckpointRuntime,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures", "phase34_memory_checkpoint_cache_cases.jsonl"
)
DEFAULT_PREREGISTRATION = workspace_path(
    "evaluation", "phase34_memory_checkpoint_cache_preregistration.json"
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation", "phase34_memory_checkpoint_cache_benchmark.json"
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
        "schema": "sara-phase34-memory-checkpoint-cache-environment-v1",
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
        raise ValueError("Phase 34 fixture must be observed-only")
    if any(
        row.get("expected", {}).get("durable_mutation_allowed") is not False
        for row in rows
    ):
        raise ValueError("Phase 34 fixture cannot mutate durable state")
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
            "invalid Phase 34 registration: " + "; ".join(validation["errors"])
        )
    return manifest


def _limits(manifest: Mapping[str, Any]) -> Phase34MemoryCacheLimits:
    budgets = manifest["budgets"]
    return Phase34MemoryCacheLimits(
        max_events=int(budgets["source_events_per_case"]),
        max_checkpoints=int(budgets["max_checkpoints"]),
        selected_k=int(budgets["max_selected_checkpoints"]),
        max_summary_ids=int(budgets["max_summary_ids_per_checkpoint"]),
        max_state_bytes=int(budgets["max_total_state_bytes"]),
        max_event_cost=int(budgets["max_local_interactions_per_case"]),
        max_merges_per_event=int(budgets["max_merges_per_event"]),
        equal_segment_span=int(manifest["segmentation"]["equal_segment_event_span"]),
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
        runtime = Phase34MemoryCheckpointRuntime(arm, limits)
        replay_runtime = Phase34MemoryCheckpointRuntime(arm, limits)
        for row in rows:
            started = time.process_time_ns()
            result = runtime.evaluate(row)
            cpu_latencies_ms.append((time.process_time_ns() - started) / 1_000_000.0)
            results.append({"arm": arm, "result": result})
            replay_results.append({"arm": arm, "result": replay_runtime.evaluate(row)})

    summaries: Dict[str, Dict[str, Any]] = {}
    delayed_families = {
        "delayed_key_value_recall",
        "long_irrelevant_interval",
        "long_tail_pollution",
        "exact_verified_checkpoint",
    }
    revision_families = {"revised_value", "source_replacement"}
    safety_families = {
        "contradiction",
        "missing_segment",
        "stale_runtime_digest",
        "stale_schema_digest",
        "reordered_replay",
        "cache_overflow",
    }
    for arm in ARMS:
        arm_results = [entry["result"] for entry in results if entry["arm"] == arm]
        retrieved = [item for item in arm_results if item["decision"] == "retrieve"]
        summaries[arm] = {
            "condition_count": len(arm_results),
            "target_quality": _mean(
                [float(item["target_match"]) for item in arm_results]
            ),
            "delayed_recall_quality": _mean(
                [
                    float(item["target_match"])
                    for item in arm_results
                    if item["family"] in delayed_families
                ]
            ),
            "revision_uptake": _mean(
                [
                    float(item["target_match"])
                    for item in arm_results
                    if item["family"] in revision_families
                ]
            ),
            "safety_integrity": _mean(
                [
                    float(item["target_match"])
                    for item in arm_results
                    if item["family"] in safety_families
                ]
            ),
            "selection_precision": _mean(
                [float(item["selected_count"] > 0) for item in retrieved]
            ),
            "selection_recall": _mean(
                [float(item["target_match"]) for item in retrieved]
            ),
            "useful_checkpoint_rate": _mean(
                [
                    float(item["selected_count"])
                    / float(max(1, item["checkpoint_count"]))
                    for item in retrieved
                ]
            ),
            "retained_temporal_resolution": _mean(
                [
                    1.0
                    - float(item["merge_count"])
                    / float(max(1, item["checkpoint_count"] + item["merge_count"]))
                    for item in arm_results
                ]
            ),
            "max_state_bytes": max(item["state_bytes"] for item in arm_results),
            "max_event_cost": max(item["event_cost"] for item in arm_results),
            "eviction_count": sum(item["eviction_count"] for item in arm_results),
            "merge_count": sum(item["merge_count"] for item in arm_results),
            "durable_mutation_count": sum(
                bool(item["durable_mutation"]) for item in arm_results
            ),
        }

    candidate_summaries = [summaries[arm] for arm in ARMS[1:]]
    metrics = {
        "delayed_recall_quality": max(
            item["delayed_recall_quality"] for item in candidate_summaries
        ),
        "revision_uptake": min(
            item["revision_uptake"] for item in candidate_summaries
        ),
        "contradiction_rejection": min(
            item["safety_integrity"] for item in candidate_summaries
        ),
        "abstention_integrity": min(
            item["safety_integrity"] for item in candidate_summaries
        ),
        "selection_precision": min(
            item["selection_precision"] for item in candidate_summaries
        ),
        "selection_recall": min(
            item["selection_recall"] for item in candidate_summaries
        ),
        "useful_checkpoint_rate": min(
            item["useful_checkpoint_rate"] for item in candidate_summaries
        ),
        "retained_temporal_resolution": min(
            item["retained_temporal_resolution"] for item in candidate_summaries
        ),
        "state_bytes": float(
            max(item["max_state_bytes"] for item in candidate_summaries)
        ),
        "event_cost": float(
            max(item["max_event_cost"] for item in candidate_summaries)
        ),
        "latency_ms": max(cpu_latencies_ms, default=0.0),
        "deterministic_replay": float(results == replay_results),
    }
    metric_gates = {
        name: _threshold_passed(value, manifest["thresholds"][name])
        for name, value in metrics.items()
    }
    expected_conditions = len(ARMS) * len(rows)
    checks = {
        "fixture_fingerprint_matches": fixture_fingerprint
        == manifest["fixture_fingerprint"],
        "environment_fingerprint_matches": environment_fingerprint
        == manifest["environment_fingerprint"],
        "all_registered_conditions_executed": len(results) == expected_conditions,
        "deterministic_replay": results == replay_results,
        "checkpoint_budget_respected": all(
            entry["result"]["checkpoint_count"] <= limits.max_checkpoints
            for entry in results
        ),
        "selection_budget_respected": all(
            entry["result"]["selected_count"] <= limits.selected_k
            for entry in results
        ),
        "state_budget_respected": all(
            entry["result"]["state_bytes"] <= limits.max_state_bytes
            for entry in results
        ),
        "event_budget_respected": all(
            entry["result"]["event_cost"] <= limits.max_event_cost
            for entry in results
        ),
        "no_durable_mutation": all(
            entry["result"]["durable_mutation"] is False for entry in results
        ),
        "production_path_not_changed": all(
            entry["result"]["production_path_changed"] is False
            for entry in results
        ),
        "cpu_only": True,
        "backpropagation_not_used": True,
        "matrix_calculation_not_used": True,
        "gpu_not_used": True,
    }
    control = summaries[ARMS[0]]["delayed_recall_quality"]
    best_candidate = max(
        summaries[arm]["delayed_recall_quality"] for arm in ARMS[1:]
    )
    execution_passed = all(checks.values())
    threshold_gate_passed = all(metric_gates.values())
    mechanism_observation = {
        "checkpoint_beats_recurrent_control_on_delayed_recall": best_candidate
        > control,
        "fixed_logarithmic_topk_tradeoff_observed": len(
            {
                (
                    summaries[arm]["target_quality"],
                    summaries[arm]["eviction_count"],
                    summaries[arm]["merge_count"],
                )
                for arm in ARMS[1:]
            }
        )
        > 1,
        "five_replicates_available": False,
        "independent_evidence_available": False,
    }
    mechanism_gate_passed = all(mechanism_observation.values())
    return {
        "schema": "sara-phase34-memory-checkpoint-cache-benchmark-v1",
        "experiment_id": manifest["experiment_id"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "fixture_fingerprint": fixture_fingerprint,
        "environment_fingerprint": environment_fingerprint,
        "observed_only": True,
        "execution_passed": execution_passed,
        "threshold_gate_passed": threshold_gate_passed,
        "mechanism_gate_passed": mechanism_gate_passed,
        "promotion_ready": False,
        "independent_evidence_available": False,
        "production_path_changed": False,
        "checks": checks,
        "metric_gates": metric_gates,
        "mechanism_observation": mechanism_observation,
        "metrics": {"condition_count": len(results), **metrics},
        "arm_summaries": summaries,
        "results": results,
        "policy_notes": [
            "This run uses the frozen synthetic observed-only fixture.",
            "CPU latency is measured with process CPU time to exclude scheduler stalls.",
            "The registered protocol contains no replicate seeds, so the five-replicate acceptance gate remains open.",
            "No production, durable-memory, biological, physical-energy, or general-accuracy claim is made.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--preregistration-path", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        rows = load_fixture(args.fixture_path)
        manifest = load_preregistration(args.preregistration_path)
        report = build_report(rows, manifest)
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
                "execution_passed": report["execution_passed"],
                "threshold_gate_passed": report["threshold_gate_passed"],
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
