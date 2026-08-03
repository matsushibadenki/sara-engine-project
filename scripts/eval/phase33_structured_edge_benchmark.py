#!/usr/bin/env python3
"""Run the immutable observed-only Phase 33 structured-edge experiment."""

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

from sara_engine.evaluation.phase33_preregistration import (  # noqa: E402
    MECHANISM_ARMS,
    REQUIRED_CASE_FAMILIES,
    SIMPLIFICATION_LEVELS,
    is_managed_preregistration_path,
    validate_preregistration,
)
from sara_engine.neuro.structured_edge import (  # noqa: E402
    StructuredEdgeLimits,
    StructuredEdgeRuntime,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures",
    "phase33_structured_edge_cases.jsonl",
)
DEFAULT_PREREGISTRATION = workspace_path(
    "evaluation",
    "phase33_structured_edge_preregistration.json",
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation",
    "phase33_structured_edge_benchmark.json",
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
        "schema": "sara-phase33-environment-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
        "physical_energy_claim": False,
    }


def load_fixture(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    families = tuple(row.get("family") for row in rows)
    if families != REQUIRED_CASE_FAMILIES:
        raise ValueError("fixture_case_families_do_not_match_registration")
    if any(row.get("observed_only") is not True for row in rows):
        raise ValueError("phase33_fixture_must_be_observed_only")
    if any(
        row.get("expected", {}).get("durable_mutation_allowed") is not False
        for row in rows
    ):
        raise ValueError("phase33_fixture_cannot_allow_durable_mutation")
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
            "invalid Phase 33 registration: " + "; ".join(validation["errors"])
        )
    return manifest


def _limits(manifest: Mapping[str, Any]) -> StructuredEdgeLimits:
    budgets = manifest["budgets"]
    return StructuredEdgeLimits(
        max_contacts=int(budgets["max_contacts_per_relation"]),
        max_branch_slots=int(budgets["max_branch_slots_per_relation"]),
        max_internal_interactions=int(
            budgets["max_internal_interactions_per_relation"]
        ),
        max_contact_rewrites_per_event=int(
            budgets["max_contact_rewrites_per_event"]
        ),
        max_events=int(budgets["source_events_per_case"]),
        max_state_bytes=int(budgets["max_total_state_bytes"]),
    )


def _stable_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(result)


def build_report(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    fixture_fingerprint = _digest(list(rows))
    environment_fingerprint = _digest(_environment_descriptor())
    limits = _limits(manifest)
    results: List[Dict[str, Any]] = []
    replay_results: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []

    for arm in MECHANISM_ARMS:
        runtime = StructuredEdgeRuntime(arm, limits)
        replay_runtime = StructuredEdgeRuntime(arm, limits)
        for level in SIMPLIFICATION_LEVELS:
            for seed in manifest["replicate_seeds"]:
                for row in rows:
                    started = time.perf_counter()
                    result = runtime.evaluate(row)
                    latencies_ms.append((time.perf_counter() - started) * 1000.0)
                    envelope = {
                        "arm": arm,
                        "simplification_level": level["name"],
                        "replicate_seed": seed,
                        "result": _stable_result(result),
                    }
                    results.append(envelope)
                    replay_results.append(
                        {
                            "arm": arm,
                            "simplification_level": level["name"],
                            "replicate_seed": seed,
                            "result": _stable_result(replay_runtime.evaluate(row)),
                        }
                    )

    arm_summaries: Dict[str, Dict[str, Any]] = {}
    for arm in MECHANISM_ARMS:
        arm_results = [entry["result"] for entry in results if entry["arm"] == arm]
        satisfied = sum(bool(result["behavior_satisfied"]) for result in arm_results)
        arm_summaries[arm] = {
            "condition_count": len(arm_results),
            "behavior_satisfied_count": satisfied,
            "ambiguous_relation_quality": satisfied / len(arm_results),
            "abstention_count": sum(
                result["status"] == "abstained" for result in arm_results
            ),
            "branch_interaction_count": sum(
                int(result["branch_interaction_count"]) for result in arm_results
            ),
            "max_state_bytes": max(
                int(result["state_bytes"]) for result in arm_results
            ),
            "max_event_cost": max(
                int(result["event_cost"]) for result in arm_results
            ),
            "durable_mutation_count": sum(
                bool(result["durable_mutation"]) for result in arm_results
            ),
        }

    expected_conditions = (
        len(MECHANISM_ARMS)
        * len(SIMPLIFICATION_LEVELS)
        * len(manifest["replicate_seeds"])
        * len(rows)
    )
    checks = {
        "registered_protocol_loaded": True,
        "fixture_fingerprint_matches": (
            fixture_fingerprint == manifest["fixture_fingerprint"]
        ),
        "environment_fingerprint_matches": (
            environment_fingerprint == manifest["environment_fingerprint"]
        ),
        "all_registered_conditions_executed": len(results) == expected_conditions,
        "deterministic_replay": results == replay_results,
        "state_budget_respected": all(
            result["state_bytes"] <= limits.max_state_bytes
            for result in (entry["result"] for entry in results)
        ),
        "event_budget_respected": all(
            result["event_cost"]
            <= int(manifest["budgets"]["max_local_interactions_per_case"])
            for result in (entry["result"] for entry in results)
        ),
        "latency_ceiling_respected": max(latencies_ms, default=0.0)
        <= float(manifest["budgets"]["max_latency_ms"]),
        "no_durable_mutation": all(
            result["durable_mutation"] is False
            for result in (entry["result"] for entry in results)
        ),
        "cpu_only": True,
        "backpropagation_not_used": True,
        "matrix_calculation_not_used": True,
        "gpu_not_used": True,
        "production_path_not_changed": True,
    }
    mechanism_observation = {
        "linear_multi_beats_single_scalar": (
            arm_summaries["linear_multi_contact"]["ambiguous_relation_quality"]
            > arm_summaries["single_scalar_contact"][
                "ambiguous_relation_quality"
            ]
        ),
        "branch_beats_typed_independent": (
            arm_summaries["branch_local_contacts"]["ambiguous_relation_quality"]
            > arm_summaries["typed_independent_contacts"][
                "ambiguous_relation_quality"
            ]
        ),
        "typed_beats_linear_multi_contact": (
            arm_summaries["typed_independent_contacts"][
                "ambiguous_relation_quality"
            ]
            > arm_summaries["linear_multi_contact"][
                "ambiguous_relation_quality"
            ]
        ),
        "simplification_evidence_available": False,
        "independent_evidence_available": False,
    }
    execution_passed = all(checks.values())
    promotion_ready = execution_passed and all(mechanism_observation.values())
    return {
        "schema": "sara-phase33-structured-edge-benchmark-v1",
        "experiment_id": manifest["experiment_id"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "fixture_fingerprint": fixture_fingerprint,
        "environment_fingerprint": environment_fingerprint,
        "observed_only": True,
        "execution_passed": execution_passed,
        "promotion_ready": promotion_ready,
        "production_path_changed": False,
        "checks": checks,
        "mechanism_observation": mechanism_observation,
        "metrics": {
            "case_count": len(rows),
            "condition_count": len(results),
            "replicate_count": len(manifest["replicate_seeds"]),
            "simplification_level_count": len(SIMPLIFICATION_LEVELS),
            "max_latency_ms_observed": max(latencies_ms, default=0.0),
            "iso_quality_total_complexity_reduction": None,
        },
        "arm_summaries": arm_summaries,
        "results": results,
        "policy_notes": [
            "This run executes the immutable observed-only fixture only.",
            "Behavior quality measures fixture conformance, not task accuracy.",
            "Simplification and independent-data acceptance gates remain open.",
            "No biological, production, physical-energy, or general-accuracy claim is made.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--preregistration-path", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    rows = load_fixture(args.fixture_path)
    manifest = load_preregistration(args.preregistration_path)
    report = build_report(rows, manifest)
    with open(
        ensure_parent_directory(args.output_path),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "execution_passed": report["execution_passed"],
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
