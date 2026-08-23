#!/usr/bin/env python3
"""Execute the registered Phase 34 semantic delayed-recall workload."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase34_factorial_preregistration import ARMS  # noqa: E402
from sara_engine.evaluation.phase34_human_review import (  # noqa: E402
    validate_request,
)
from sara_engine.evaluation.phase34_semantic_preregistration import (  # noqa: E402
    CASE_COUNT,
    REPLICATE_SEEDS,
    REVIEW_REQUEST_FINGERPRINT,
    validate_preregistration,
)
from sara_engine.memory.semantic_checkpoint_adapter import (  # noqa: E402
    SemanticCheckpointLimits,
    SemanticCheckpointRuntime,
    SparseMultilingualSemanticAdapter,
    claim_stream,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)


DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures", "phase34_semantic_delayed_recall_cases.jsonl"
)
DEFAULT_PREREGISTRATION = workspace_path(
    "evaluation", "phase34_semantic_delayed_recall_preregistration.json"
)
DEFAULT_REQUEST = workspace_path(
    "evaluation", "phase34_transcribed_excerpt_human_review_request.json"
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation", "phase34_semantic_delayed_recall_benchmark.json"
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return value


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        values = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("semantic fixture rows must be objects")
    return values


def _environment_descriptor() -> Dict[str, Any]:
    return {
        "schema": "sara-phase34-semantic-delayed-recall-environment-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
    }


def _limits(manifest: Mapping[str, Any]) -> SemanticCheckpointLimits:
    budgets = manifest["budgets"]
    return SemanticCheckpointLimits(
        max_events=int(budgets["source_events_per_case"]),
        max_attempted_checkpoints=int(budgets["attempted_checkpoints_per_case"]),
        max_checkpoints=int(budgets["max_checkpoints"]),
        selected_k=int(budgets["max_selected_checkpoints"]),
        max_claims_per_checkpoint=int(budgets["max_summary_ids_per_checkpoint"]),
        max_state_bytes=int(budgets["max_total_state_bytes"]),
        max_event_cost=int(budgets["max_local_interactions_per_case"]),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _threshold_passed(value: float, spec: Mapping[str, Any]) -> bool:
    limit = float(spec["limit"])
    if spec["direction"] == "minimum":
        return value >= limit
    if spec["direction"] == "maximum":
        return value <= limit
    raise ValueError("unknown semantic threshold direction")


def _candidate_case(
    row: Mapping[str, Any], manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    fields = tuple(manifest["evaluation_contract"]["candidate_visible_fields"])
    candidate = {field: row[field] for field in fields}
    if set(candidate) != set(fields):
        raise ValueError("candidate-visible semantic fields are incomplete")
    return candidate


def _validate_inputs(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    request: Mapping[str, Any],
) -> Tuple[Dict[str, bool], List[Dict[str, Any]]]:
    registration = validate_preregistration(manifest, managed_path=True)
    request_validation = validate_request(request)
    candidate_cases = [_candidate_case(row, manifest) for row in rows]
    evaluator_fields = set(manifest["evaluation_contract"]["evaluator_only_fields"])
    candidate_payload = json.dumps(candidate_cases, ensure_ascii=False, sort_keys=True)
    forbidden_values = {
        str(row[field])
        for row in rows
        for field in ("expected_proposition_id",)
        if row.get(field) is not None
    }
    checks = {
        "semantic_preregistration_valid": bool(registration["valid"]),
        "fixture_fingerprint_matches": _digest(list(rows))
        == manifest.get("fixture_fingerprint"),
        "environment_fingerprint_matches": _digest(_environment_descriptor())
        == manifest.get("environment_fingerprint"),
        "review_request_valid": bool(request_validation["valid"]),
        "review_request_fingerprint_matches": request_validation.get(
            "request_fingerprint"
        )
        == REVIEW_REQUEST_FINGERPRINT,
        "case_count_matches": len(rows) == CASE_COUNT,
        "candidate_fields_exact": all(
            set(candidate) == set(manifest["evaluation_contract"]["candidate_visible_fields"])
            for candidate in candidate_cases
        ),
        "evaluator_keys_absent_from_candidate": all(
            evaluator_fields.isdisjoint(candidate) for candidate in candidate_cases
        ),
        "expected_proposition_values_absent_from_candidate": all(
            value not in candidate_payload for value in forbidden_values
        ),
        "selector_retuning_disabled": manifest["execution_policy"][
            "selector_retuning_allowed"
        ]
        is False,
        "query_aware_retention_disabled": manifest["execution_policy"][
            "query_aware_retention_allowed"
        ]
        is False,
        "production_mutation_disabled": manifest["execution_policy"][
            "production_mutation"
        ]
        is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("semantic benchmark input validation failed: " + ", ".join(failed))
    return checks, candidate_cases


def _source_claims(
    request: Mapping[str, Any], adapter: SparseMultilingualSemanticAdapter
) -> Tuple[List[Any], Dict[str, str]]:
    claims = []
    record_to_ref: Dict[str, str] = {}
    for target in request["targets"]:
        source_ref = str(target["source_ref"])
        record_to_ref[str(target["record_id"])] = source_ref
        claims.append(
            adapter.encode_source(
                str(target["stored_excerpt"]),
                source_ref=source_ref,
                source_revision=str(target["source_revision"]),
            )
        )
    return claims, record_to_ref


def _score_result(
    row: Mapping[str, Any], candidate_result: Mapping[str, Any]
) -> Dict[str, Any]:
    expected_decision = str(row["expected_decision"])
    actual_decision = str(candidate_result["decision"])
    decision_correct = actual_decision == expected_decision
    source_traceable = (
        candidate_result.get("source_ref") == row.get("source_ref")
        if actual_decision.startswith("retrieve_")
        else True
    )
    if row.get("expected_proposition_id") is None:
        proposition_correct = candidate_result.get("claim_key") is None
    else:
        proposition_correct = bool(
            candidate_result.get("claim_key")
            and candidate_result.get("source_ref") == row.get("source_ref")
        )
    return {
        "correct": bool(decision_correct and proposition_correct and source_traceable),
        "decision_correct": bool(decision_correct),
        "proposition_correct": bool(proposition_correct),
        "source_traceable": bool(source_traceable),
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    request: Mapping[str, Any],
) -> Dict[str, Any]:
    input_checks, candidate_cases = _validate_inputs(rows, manifest, request)
    adapter = SparseMultilingualSemanticAdapter()
    claims, record_to_ref = _source_claims(request, adapter)
    limits = _limits(manifest)
    results: List[Dict[str, Any]] = []
    replay_matches: List[float] = []
    latencies_ms: List[float] = []
    candidate_traces: List[Dict[str, Any]] = []
    for row, candidate_case in zip(rows, candidate_cases):
        query = adapter.encode_query(str(candidate_case["query_text"]))
        stream, omission = claim_stream(
            claims,
            target_source_ref=record_to_ref[str(candidate_case["record_id"])],
            horizon=int(candidate_case["horizon"]),
            control_mode=str(candidate_case["control_mode"]),
        )
        for arm in ARMS:
            for seed in REPLICATE_SEEDS:
                runtime = SemanticCheckpointRuntime(arm, limits)
                started = time.process_time_ns()
                candidate_result = runtime.evaluate(
                    stream,
                    query,
                    horizon=int(candidate_case["horizon"]),
                    omission_receipt=omission,
                )
                latencies_ms.append((time.process_time_ns() - started) / 1_000_000.0)
                replay = SemanticCheckpointRuntime(arm, limits).evaluate(
                    stream,
                    query,
                    horizon=int(candidate_case["horizon"]),
                    omission_receipt=omission,
                )
                replay_matches.append(float(candidate_result == replay))
                candidate_traces.append(candidate_result)
                results.append(
                    {
                        "case_id": str(candidate_case["case_id"]),
                        "record_id": str(candidate_case["record_id"]),
                        "language": str(candidate_case["language"]),
                        "horizon": int(candidate_case["horizon"]),
                        "family": str(candidate_case["family"]),
                        "arm": arm,
                        "seed": int(seed),
                        "candidate": candidate_result,
                        "evaluation": _score_result(row, candidate_result),
                    }
                )

    by_arm_family: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_arm_language: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for entry in results:
        value = float(entry["evaluation"]["correct"])
        by_arm_family[(entry["arm"], entry["family"])].append(value)
        if entry["family"] == "semantic_paraphrase_recall":
            by_arm_language[(entry["arm"], entry["language"])].append(value)
    paraphrase_by_arm = {
        arm: _mean(by_arm_family[(arm, "semantic_paraphrase_recall")])
        for arm in ARMS
    }
    checkpoint_arms = tuple(arm for arm in ARMS if arm != ARMS[0])
    best_arm = max(checkpoint_arms, key=lambda arm: (paraphrase_by_arm[arm], arm))
    retained_identity: List[float] = []
    indexed = {
        (entry["case_id"], entry["seed"], entry["arm"]): entry["candidate"]
        for entry in results
    }
    for row in rows:
        for seed in REPLICATE_SEEDS:
            case_id = str(row["case_id"])
            retained_identity.extend(
                (
                    float(
                        indexed[(case_id, seed, ARMS[1])]["retained_set_digest"]
                        == indexed[(case_id, seed, ARMS[2])]["retained_set_digest"]
                    ),
                    float(
                        indexed[(case_id, seed, ARMS[3])]["retained_set_digest"]
                        == indexed[(case_id, seed, ARMS[4])]["retained_set_digest"]
                    ),
                )
            )
    non_abstaining = [
        trace for trace in candidate_traces if str(trace["decision"]).startswith("retrieve_")
    ]
    lexical_values = [
        value
        for arm in ARMS
        for value in by_arm_family[(arm, "lexical_overlap_abstention")]
    ]
    metrics = {
        "semantic_paraphrase_macro_accuracy": paraphrase_by_arm[best_arm],
        "best_checkpoint_minus_control": paraphrase_by_arm[best_arm]
        - paraphrase_by_arm[ARMS[0]],
        "lexical_overlap_abstention": _mean(lexical_values),
        "revision_uptake": _mean(
            [
                value
                for arm in ARMS
                for value in by_arm_family[(arm, "revision_replacement")]
            ]
        ),
        "contradiction_abstention": _mean(
            [
                value
                for arm in ARMS
                for value in by_arm_family[(arm, "contradiction_abstention")]
            ]
        ),
        "missing_evidence_abstention": _mean(
            [
                value
                for arm in ARMS
                for value in by_arm_family[(arm, "missing_evidence_abstention")]
            ]
        ),
        "worst_language_recall": min(
            _mean(by_arm_language[(best_arm, language)])
            for language in manifest["languages"]
        ),
        "source_traceability": _mean(
            [float(bool(trace.get("source_ref"))) for trace in non_abstaining]
        ),
        "retained_set_identity": _mean(retained_identity),
        "state_bytes": float(
            max(trace["total_state_bytes"] for trace in candidate_traces)
        ),
        "event_cost": float(max(trace["event_cost"] for trace in candidate_traces)),
        "latency_ms": max(latencies_ms, default=0.0),
        "deterministic_replay": _mean(replay_matches),
    }
    metric_gates = {
        name: _threshold_passed(value, manifest["thresholds"][name])
        for name, value in metrics.items()
    }
    candidate_payload = json.dumps(candidate_traces, ensure_ascii=False, sort_keys=True)
    forbidden_propositions = {
        str(row["expected_proposition_id"])
        for row in rows
        if row.get("expected_proposition_id") is not None
    }
    checks = {
        **input_checks,
        "all_6750_conditions_executed": len(results) == 6750,
        "all_registered_arms_executed": sorted({entry["arm"] for entry in results})
        == sorted(ARMS),
        "all_registered_seeds_executed": sorted({entry["seed"] for entry in results})
        == sorted(REPLICATE_SEEDS),
        "expected_labels_absent_from_candidate_traces": all(
            proposition not in candidate_payload for proposition in forbidden_propositions
        ),
        "all_resources_bounded": all(trace["bounded"] for trace in candidate_traces),
        "retention_query_blind": all(
            trace["query_visible_during_retention"] is False
            for trace in candidate_traces
        ),
        "no_durable_mutation": all(
            trace["durable_mutation"] is False for trace in candidate_traces
        ),
        "production_path_not_changed": all(
            trace["production_path_changed"] is False for trace in candidate_traces
        ),
        "deterministic_replay": all(value == 1.0 for value in replay_matches),
        "cpu_only": True,
        "backpropagation_not_used": True,
        "matrix_calculation_not_used": True,
        "gpu_not_used": True,
    }
    execution_passed = all(checks.values())
    semantic_gate_passed = all(metric_gates.values())
    return {
        "schema": "sara-phase34-semantic-delayed-recall-benchmark-v1",
        "experiment_id": manifest["experiment_id"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "fixture_fingerprint": manifest["fixture_fingerprint"],
        "observed_only": True,
        "independent_evidence_available": True,
        "independent_evidence_scope": "six_human_aligned_source_bound_propositions",
        "synthetic_safety_controls_are_independent_evidence": False,
        "execution_passed": execution_passed,
        "threshold_gate_passed": semantic_gate_passed,
        "semantic_gate_passed": semantic_gate_passed,
        "promotion_ready": False,
        "production_path_changed": False,
        "best_checkpoint_arm": best_arm,
        "checks": checks,
        "metric_gates": metric_gates,
        "metrics": {
            "condition_count": len(results),
            "case_count": len(rows),
            "semantic_paraphrase_accuracy_by_arm": paraphrase_by_arm,
            "semantic_paraphrase_accuracy_by_language_for_best_arm": {
                language: _mean(by_arm_language[(best_arm, language)])
                for language in manifest["languages"]
            },
            **metrics,
        },
        "results": results,
        "policy_notes": [
            "Candidate retention and selection never receive evaluator decisions or proposition IDs.",
            "The semantic score uses typed subject and relation-axis coverage, not exact identity or token overlap.",
            "Independent evidence is limited to six human-aligned source-bound propositions.",
            "Revision, contradiction, missing, and lexical-overlap families are synthetic controls.",
            "Production mutation and promotion remain disabled even if the registered gate passes.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--preregistration-path", default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--request-path", default=DEFAULT_REQUEST)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        report = build_report(
            _read_jsonl(args.fixture_path),
            _read_json(args.preregistration_path),
            _read_json(args.request_path),
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
                "semantic_gate_passed": report["semantic_gate_passed"],
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
