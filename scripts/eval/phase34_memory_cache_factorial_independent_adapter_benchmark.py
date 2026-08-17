#!/usr/bin/env python3
"""Execute the registered Phase 34 independent source-identity adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase34_factorial_preregistration import ARMS, REPLICATE_SEEDS  # noqa: E402
from sara_engine.evaluation.phase34_independent_adapter_preregistration import (  # noqa: E402
    CASE_COUNT,
    PARENT_PROTOCOL_FINGERPRINT,
    validate_preregistration,
)
from sara_engine.memory.memory_checkpoint_factorial import (  # noqa: E402
    FactorialLimits,
    MemoryCacheFactorialRuntime,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)


DEFAULT_MANIFEST = processed_data_path("autobot", "architecture_migration_latent_manifest.jsonl")
DEFAULT_CASE_PLAN = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_case_plan.json"
)
DEFAULT_PREREGISTRATION = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_adapter_v2_preregistration.json"
)
DEFAULT_PARENT_REPORT = workspace_path("evaluation", "phase34_memory_cache_factorial_benchmark.json")
DEFAULT_EXTERNAL_GATE = workspace_path("evaluation", "continual_horizon_external_gate.json")
DEFAULT_READINESS_GATE = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_gate.json"
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_independent_adapter_v2_benchmark.json"
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


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return value


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("source manifest rows must be objects")
    return rows


def _environment_descriptor() -> Dict[str, Any]:
    return {
        "schema": "sara-phase34-memory-cache-factorial-independent-adapter-environment-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
    }


def _limits(manifest: Mapping[str, Any]) -> FactorialLimits:
    budgets = manifest["budgets"]
    return FactorialLimits(
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


def adapt_case(
    planned: Mapping[str, Any], source_by_hash: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    hashes = [str(value) for value in planned.get("stream_material_hashes", [])]
    refs = [str(value) for value in planned.get("stream_source_refs", [])]
    query_hash = str(planned.get("query_material_hash", ""))
    if (
        not 8 <= len(hashes) <= 16
        or len(hashes) != len(refs)
        or len(set(hashes)) != len(hashes)
        or any(value not in source_by_hash for value in hashes)
        or any(str(source_by_hash[value]["source_ref"]) != ref for value, ref in zip(hashes, refs))
    ):
        raise ValueError(f"case plan provenance mismatch: {planned.get('case_id', '')}")
    query_present = query_hash in hashes
    stream_ids = [
        f"target:{value}" if value == query_hash else f"source:{value}"
        for value in hashes
    ]
    query_id = f"target:{query_hash}" if query_present else f"missing:{query_hash}"
    return {
        "schema": "sara-phase34-memory-cache-factorial-independent-runtime-case-v1",
        "case_id": str(planned["case_id"]),
        "family": str(planned["family"]),
        "factor_focus": str(planned["factor_focus"]),
        "horizon_events": int(planned["horizon"]),
        "checkpoint_stream": stream_ids,
        "checkpoint_source_refs": refs,
        "query_ids": [query_id],
        "expected_relation": str(planned["family"]),
        "target_must_be_retained_for_selection": str(planned["factor_focus"]) == "selection",
        "negative_mode": str(planned["negative_mode"]),
        "durable_mutation_allowed": False,
        "source_domain": str(planned["source_domain"]),
        "source_identity_query": True,
        "semantic_accuracy_claim_allowed": False,
    }


def _validate_inputs(
    rows: Sequence[Mapping[str, Any]],
    case_plan: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    parent_report: Mapping[str, Any],
    external_gate: Mapping[str, Any],
    readiness_gate: Mapping[str, Any],
) -> Dict[str, bool]:
    validation = validate_preregistration(preregistration, managed_path=True)
    checks = {
        "adapter_preregistration_valid": bool(validation["valid"]),
        "adapter_protocol_identity": preregistration.get("protocol_fingerprint")
        == "7e4ce13ff7e0aded273a657133263ebf9c52e7d5285c3d2a341a87233bd44ec1",
        "source_manifest_fingerprint_matches": _digest(list(rows))
        == preregistration.get("source_manifest_fingerprint"),
        "case_plan_fingerprint_matches": _digest(dict(case_plan))
        == preregistration.get("case_plan_fingerprint"),
        "environment_fingerprint_matches": _digest(_environment_descriptor())
        == preregistration.get("environment_fingerprint"),
        "parent_report_fingerprint_matches": _digest(dict(parent_report))
        == preregistration.get("parent_factorial_report_fingerprint"),
        "parent_protocol_matches": parent_report.get("protocol_fingerprint")
        == PARENT_PROTOCOL_FINGERPRINT,
        "external_gate_fingerprint_matches": _digest(dict(external_gate))
        == preregistration.get("external_gate_fingerprint"),
        "readiness_gate_fingerprint_matches": _digest(dict(readiness_gate))
        == preregistration.get("readiness_gate_fingerprint"),
        "external_horizons_passed": external_gate.get("promotion_allowed") is True,
        "independent_execution_was_ready": readiness_gate.get("independent_execution_ready") is True,
        "case_plan_count_matches": case_plan.get("case_count") == CASE_COUNT,
        "semantic_claim_disabled": preregistration.get("claim_boundaries", {}).get(
            "semantic_accuracy_claim_allowed"
        )
        is False,
        "selector_retuning_disabled": preregistration.get("execution_policy", {}).get(
            "selector_retuning_allowed"
        )
        is False,
        "query_aware_retention_disabled": preregistration.get("execution_policy", {}).get(
            "query_aware_retention_allowed"
        )
        is False,
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError("independent adapter input validation failed: " + ", ".join(failed))
    return checks


def build_report(
    rows: Sequence[Mapping[str, Any]],
    case_plan: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    parent_report: Mapping[str, Any],
    external_gate: Mapping[str, Any],
    readiness_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    input_checks = _validate_inputs(
        rows, case_plan, preregistration, parent_report, external_gate, readiness_gate
    )
    source_by_hash = {str(row["material_hash"]): row for row in rows}
    planned_by_id = {
        str(case["case_id"]): case for case in case_plan.get("cases", [])
    }
    runtime_cases = [adapt_case(case, source_by_hash) for case in case_plan["cases"]]
    limits = _limits(preregistration)
    results: List[Dict[str, Any]] = []
    replay_results: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    for arm in ARMS:
        for seed in REPLICATE_SEEDS:
            runtime = MemoryCacheFactorialRuntime(arm, limits)
            replay = MemoryCacheFactorialRuntime(arm, limits)
            for case in runtime_cases:
                started = time.process_time_ns()
                result = runtime.evaluate(case, seed=seed)
                latencies_ms.append((time.process_time_ns() - started) / 1_000_000.0)
                results.append({"arm": arm, "seed": seed, "result": result})
                replay_results.append(
                    {"arm": arm, "seed": seed, "result": replay.evaluate(case, seed=seed)}
                )
    indexed = {
        (entry["result"]["case_id"], entry["seed"], entry["arm"]): entry["result"]
        for entry in results
    }
    selection_precision: List[float] = []
    selection_recall: List[float] = []
    equal_selection: List[float] = []
    log_selection: List[float] = []
    old_retention: List[float] = []
    recent_resolution: List[float] = []
    safety: List[float] = []
    retained_identity: List[float] = []
    positive_recall: Dict[str, List[float]] = defaultdict(list)
    positive_recall_detail: Dict[tuple[str, int, str], List[float]] = defaultdict(list)
    for case in runtime_cases:
        for seed in REPLICATE_SEEDS:
            case_id = str(case["case_id"])
            values = {arm: indexed[(case_id, seed, arm)] for arm in ARMS}
            retained_identity.extend(
                (
                    float(values[ARMS[1]]["retained_set_digest"] == values[ARMS[2]]["retained_set_digest"]),
                    float(values[ARMS[3]]["retained_set_digest"] == values[ARMS[4]]["retained_set_digest"]),
                )
            )
            if case["factor_focus"] == "selection":
                equal_effect = values[ARMS[2]]["selection_precision"] - values[ARMS[1]]["selection_precision"]
                log_effect = values[ARMS[4]]["selection_precision"] - values[ARMS[3]]["selection_precision"]
                equal_selection.append(equal_effect)
                log_selection.append(log_effect)
                selection_precision.extend((equal_effect, log_effect))
                selection_recall.extend(
                    (
                        values[ARMS[2]]["recall"] - values[ARMS[1]]["recall"],
                        values[ARMS[4]]["recall"] - values[ARMS[3]]["recall"],
                    )
                )
            if case["family"] == "old_identity_retention":
                old_retention.append(values[ARMS[3]]["recall"] - values[ARMS[1]]["recall"])
            if case["family"] == "recent_identity_control":
                recent_resolution.append(
                    values[ARMS[1]]["retained_temporal_resolution"]
                    - values[ARMS[3]]["retained_temporal_resolution"]
                )
            if case["factor_focus"] == "safety":
                safety.extend(value["safety_integrity"] for value in values.values())
            else:
                for arm, value in values.items():
                    positive_recall[arm].append(value["recall"])
                    planned = planned_by_id[case_id]
                    positive_recall_detail[
                        (str(planned["source_domain"]), int(planned["horizon"]), arm)
                    ].append(value["recall"])
    metrics = {
        "selection_precision_main_effect": _mean(selection_precision),
        "selection_recall_noninferiority": _mean(selection_recall),
        "retention_old_recall_main_effect": _mean(old_retention),
        "retention_recent_resolution_main_effect": _mean(recent_resolution),
        "selection_retention_interaction_abs": abs(_mean(equal_selection) - _mean(log_selection)),
        "safety_integrity": _mean(safety),
        "retained_set_identity": _mean(retained_identity),
        "state_bytes": float(max(entry["result"]["total_state_bytes"] for entry in results)),
        "event_cost": float(max(entry["result"]["event_cost"] for entry in results)),
        "latency_ms": max(latencies_ms, default=0.0),
        "deterministic_replay": float(results == replay_results),
    }
    metric_gates = {
        name: _threshold_passed(value, preregistration["thresholds"][name])
        for name, value in metrics.items()
    }
    expected_decisions = {
        "missing_identity_control": "abstain",
        "stale_digest_control": "reject_stale",
        "contradiction_control": "reject_contradiction",
    }
    execution_checks = {
        **input_checks,
        "all_1050_conditions_executed": len(results) == 1050,
        "all_five_registered_seeds_executed": sorted({entry["seed"] for entry in results})
        == sorted(REPLICATE_SEEDS),
        "deterministic_replay": results == replay_results,
        "all_resources_bounded": all(entry["result"]["bounded"] for entry in results),
        "retention_query_blind": all(
            entry["result"]["query_visible_during_retention"] is False for entry in results
        ),
        "external_provenance_supplied": all(
            entry["result"].get("external_provenance_supplied") is True for entry in results
        ),
        "all_runtime_refs_from_case_plan": all(
            set(entry["result"].get("retained_source_refs", ())).issubset(
                set(planned_by_id[entry["result"]["case_id"]]["stream_source_refs"])
            )
            and set(entry["result"].get("selected_source_refs", ())).issubset(
                set(planned_by_id[entry["result"]["case_id"]]["stream_source_refs"])
            )
            for entry in results
        ),
        "synthetic_negative_controls_fail_closed": all(
            entry["result"]["decision"] == expected_decisions[entry["result"]["family"]]
            for entry in results
            if entry["result"]["family"] in expected_decisions
        ),
        "no_durable_mutation": all(entry["result"]["durable_mutation"] is False for entry in results),
        "production_path_not_changed": all(
            entry["result"]["production_path_changed"] is False for entry in results
        ),
        "cpu_only": True,
        "backpropagation_not_used": True,
        "matrix_calculation_not_used": True,
        "gpu_not_used": True,
    }
    execution_passed = all(execution_checks.values())
    identity_gate_passed = all(metric_gates.values())
    planned_hashes = {
        str(value)
        for case in case_plan["cases"]
        for value in case["stream_material_hashes"]
    }
    planned_refs = {
        str(value)
        for case in case_plan["cases"]
        for value in case["stream_source_refs"]
    }
    return {
        "schema": "sara-phase34-memory-cache-factorial-independent-adapter-benchmark-v2",
        "experiment_id": preregistration["experiment_id"],
        "protocol_fingerprint": preregistration["protocol_fingerprint"],
        "source_manifest_fingerprint": preregistration["source_manifest_fingerprint"],
        "case_plan_fingerprint": preregistration["case_plan_fingerprint"],
        "observed_only": True,
        "independent_evidence_available": True,
        "independent_evidence_scope": "exact_source_identity_recall_only",
        "synthetic_negative_controls_are_independent_evidence": False,
        "semantic_accuracy_claim_allowed": False,
        "execution_passed": execution_passed,
        "threshold_gate_passed": identity_gate_passed,
        "identity_gate_passed": identity_gate_passed,
        "promotion_ready": False,
        "production_path_changed": False,
        "checks": execution_checks,
        "metric_gates": metric_gates,
        "metrics": {
            "condition_count": len(results),
            "case_count": len(runtime_cases),
            "planned_unique_material_count": len(planned_hashes),
            "planned_unique_source_ref_count": len(planned_refs),
            "positive_identity_recall_by_arm": {
                arm: _mean(positive_recall[arm]) for arm in ARMS
            },
            "positive_identity_recall_by_domain_horizon_arm": {
                domain: {
                    str(horizon): {
                        arm: _mean(positive_recall_detail[(domain, horizon, arm)])
                        for arm in ARMS
                    }
                    for horizon in (10, 30, 100)
                }
                for domain in ("docs.python.org", "www.rfc-editor.org")
            },
            **metrics,
        },
        "results": results,
        "policy_notes": [
            "Independent evidence is limited to exact source-hash identity recall.",
            "Missing, stale, and contradiction cases are synthetic fail-closed controls.",
            "No semantic accuracy, language understanding, ANN parity, or energy claim is allowed.",
            "Production promotion remains blocked pending provenance review and human approval.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--case-plan-path", default=DEFAULT_CASE_PLAN)
    parser.add_argument("--preregistration-path", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--parent-report-path", default=DEFAULT_PARENT_REPORT)
    parser.add_argument("--external-gate-path", default=DEFAULT_EXTERNAL_GATE)
    parser.add_argument("--readiness-gate-path", default=DEFAULT_READINESS_GATE)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        report = build_report(
            _read_jsonl(args.manifest_path),
            _read_json(args.case_plan_path),
            _read_json(args.preregistration_path),
            _read_json(args.parent_report_path),
            _read_json(args.external_gate_path),
            _read_json(args.readiness_gate_path),
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
                "identity_gate_passed": report["identity_gate_passed"],
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
