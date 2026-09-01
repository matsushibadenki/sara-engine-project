"""Evaluator-isolated benchmark execution for Phase 30 temporal controls."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from time import process_time_ns
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .phase30_fixtures import digest as fixture_digest
from .phase30_fixtures import validate_fixtures
from .phase30_preregistration import ARMS, CASE_FAMILIES
from .phase30_runtime import run_control


BENCHMARK_SCHEMA = "sara-phase30-temporal-effective-interaction-benchmark-v1"
DECISION_SCHEMA = "sara-phase30-temporal-decision-freeze-v1"
TIMING_PERTURBATIONS = frozenset({"shuffled_time", "phase_shifted"})
INVALIDATION_FAMILIES = frozenset({"context_revision", "stale_cache", "contradiction"})


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def freeze_decisions(inputs: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run candidates without accepting evaluator labels."""
    decisions: List[Dict[str, Any]] = []
    for case in inputs:
        for arm in ARMS:
            started = process_time_ns()
            result = run_control(case, arm)
            latency_ms = (process_time_ns() - started) / 1_000_000.0
            decisions.append(
                {
                    "case_id": str(case["case_id"]),
                    "partition": str(case["partition"]),
                    "case_family": str(case["case_family"]),
                    "replicate_seed": int(case["replicate_seed"]),
                    "arm": arm,
                    "decision": result["decision"],
                    "score": result["score"],
                    "event_count": result["event_count"],
                    "event_cost": result["event_cost"],
                    "state_bytes": result["state_bytes"],
                    "cache_bytes": result["cache_bytes"],
                    "active_edge_count": result["active_edge_count"],
                    "cached_interaction_count": result["cached_interaction_count"],
                    "cache_hits": result["cache_hits"],
                    "cache_builds": result["cache_builds"],
                    "direct_computes": result["direct_computes"],
                    "invalidation_event_cost": len(result["invalidation_trace"]),
                    "invalidation_trace": result["invalidation_trace"],
                    "cpu_latency_ms": round(latency_ms, 9),
                    "runtime_replay_digest": result["replay_digest"],
                }
            )
    identity = {
        "schema": DECISION_SCHEMA,
        "decision_count": len(decisions),
        "input_digest": fixture_digest(list(inputs)),
        "decision_digest": _digest(decisions),
        "evaluator_key_loaded": False,
        "production_mutation": False,
    }
    return decisions, identity


def _replay_equivalence(inputs: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]]) -> float:
    expected = {(str(row["case_id"]), str(row["arm"])): str(row["runtime_replay_digest"]) for row in decisions}
    matches = 0
    total = 0
    for case in inputs:
        for arm in ARMS:
            total += 1
            if run_control(case, arm)["replay_digest"] == expected[(str(case["case_id"]), arm)]:
                matches += 1
    return matches / total if total else 0.0


def _recovery_events(row: Mapping[str, Any]) -> int:
    invalidations = [int(item["order"]) for item in row["invalidation_trace"] if item["reason"] in {"context_revision", "contradiction", "expiry"}]
    if not invalidations:
        return 0
    last_invalidation = max(invalidations)
    later = [int(item["order"]) for item in row["invalidation_trace"] if int(item["order"]) > last_invalidation and item["cache_entry_removed"]]
    return 0 if not later else min(later) - last_invalidation


def _metric_summary(rows: Sequence[Mapping[str, Any]], key_by_id: Mapping[str, Mapping[str, Any]], deterministic_replay: float) -> Dict[str, Any]:
    scored = []
    for row in rows:
        key = key_by_id[str(row["case_id"])]
        correct = str(row["decision"]) == str(key["expected_decision"])
        confidence = 0.0 if row["decision"] == "abstain" else min(1.0, abs(float(row["score"])))
        scored.append((row, key, correct, confidence))
    timing = [item for item in scored if item[1]["timing_required"]]
    expected_abstain = [item for item in scored if item[1]["expected_decision"] == "abstain"]
    perturbations = [item for item in scored if item[0]["case_family"] in TIMING_PERTURBATIONS]
    invalidations = [item for item in scored if item[0]["case_family"] in INVALIDATION_FAMILIES]
    reuse_denominator = sum(int(item[0]["cache_hits"]) + int(item[0]["direct_computes"]) for item in scored)
    useful_reuse_cases = [item for item in scored if int(item[0]["cache_hits"]) > 0]
    calibration = 1.0 - sum(abs(item[3] - (1.0 if item[2] else 0.0)) for item in scored) / len(scored)
    return {
        "case_count": len(scored),
        "accuracy": sum(item[2] for item in scored) / len(scored),
        "timing_required_accuracy": sum(item[2] for item in timing) / len(timing),
        "calibration": calibration,
        "justified_abstention": sum(item[2] for item in expected_abstain) / len(expected_abstain),
        "timing_perturbation_abstention": sum(item[0]["decision"] == "abstain" for item in perturbations) / len(perturbations),
        "revision_recovery_events": max((_recovery_events(item[0]) for item in invalidations), default=0),
        "stale_cache_harm": sum(not item[2] and item[0]["decision"] != "abstain" for item in invalidations) / len(invalidations),
        "cache_hit_rate": sum(int(item[0]["cache_hits"]) for item in scored) / reuse_denominator if reuse_denominator else 0.0,
        "useful_reuse_rate": sum(item[2] for item in useful_reuse_cases) / len(useful_reuse_cases) if useful_reuse_cases else 0.0,
        "construction_event_cost": max((2 * int(item[0]["cache_builds"]) for item in scored), default=0),
        "invalidation_event_cost": max((int(item[0]["invalidation_event_cost"]) for item in scored), default=0),
        "deterministic_replay": deterministic_replay,
        "max_event_cost": max(int(item[0]["event_cost"]) for item in scored),
        "max_state_bytes": max(int(item[0]["state_bytes"]) for item in scored),
        "max_cache_bytes": max(int(item[0]["cache_bytes"]) for item in scored),
        "max_active_edges": max(int(item[0]["active_edge_count"]) for item in scored),
        "max_cached_interactions": max(int(item[0]["cached_interaction_count"]) for item in scored),
        "max_cpu_latency_ms": max(float(item[0]["cpu_latency_ms"]) for item in scored),
    }


def _passes(value: float, rule: Mapping[str, Any]) -> bool:
    if rule["direction"] == "minimum":
        return value >= float(rule["limit"])
    return value <= float(rule["limit"])


def evaluate_frozen_decisions(
    inputs: Sequence[Mapping[str, Any]],
    evaluator_keys: Sequence[Mapping[str, Any]],
    fixture_manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    decision_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_fixtures(inputs, evaluator_keys, fixture_manifest)
    if decision_identity.get("evaluator_key_loaded") is not False:
        raise ValueError("decision_freeze_not_evaluator_isolated")
    if decision_identity.get("decision_digest") != _digest(list(decisions)):
        raise ValueError("decision_digest_mismatch")
    if decision_identity.get("input_digest") != fixture_manifest.get("input_digest"):
        raise ValueError("decision_input_digest_mismatch")
    expected_count = len(inputs) * len(ARMS)
    if len(decisions) != expected_count:
        raise ValueError("decision_count_mismatch")
    pairs = [(str(row["case_id"]), str(row["arm"])) for row in decisions]
    if len(set(pairs)) != expected_count:
        raise ValueError("duplicate_or_missing_decision_identity")

    key_by_id = {str(row["case_id"]): row for row in evaluator_keys}
    evaluation_rows = [row for row in decisions if row["partition"] == "evaluation"]
    replay = _replay_equivalence(inputs, decisions)
    summaries = {
        arm: _metric_summary([row for row in evaluation_rows if row["arm"] == arm], key_by_id, replay)
        for arm in ARMS
    }
    timing_best_control = max(summaries[arm]["timing_required_accuracy"] for arm in ARMS[:-1])
    timing_delta = summaries[ARMS[-1]]["timing_required_accuracy"] - timing_best_control
    summaries[ARMS[-1]]["timing_sensitivity_delta"] = timing_delta
    for arm in ARMS[:-1]:
        summaries[arm]["timing_sensitivity_delta"] = summaries[arm]["timing_required_accuracy"] - summaries["history_averaged_static_interaction"]["timing_required_accuracy"]

    thresholds = preregistration["thresholds"]
    threshold_results: Dict[str, Dict[str, Any]] = {}
    candidate = summaries[ARMS[-1]]
    for name, rule in thresholds.items():
        value = float(candidate[name])
        threshold_results[name] = {"value": value, "direction": rule["direction"], "limit": rule["limit"], "passed": _passes(value, rule)}

    budgets = preregistration["budgets"]
    budget_values = {
        "max_active_edges": candidate["max_active_edges"],
        "max_recent_events_per_edge": 16,
        "max_cached_interactions": candidate["max_cached_interactions"],
        "max_state_bytes": candidate["max_state_bytes"],
        "max_cache_bytes": candidate["max_cache_bytes"],
        "max_event_cost": candidate["max_event_cost"],
        "max_cpu_latency_ms": candidate["max_cpu_latency_ms"],
    }
    budget_results = {
        name: {"value": value, "limit": budgets[name], "passed": value <= budgets[name]}
        for name, value in budget_values.items()
    }
    comparative = preregistration["comparative_acceptance"]
    cache_accuracy = candidate["timing_required_accuracy"]
    best_control_accuracy = max(summaries[arm]["timing_required_accuracy"] for arm in ARMS[:-1])
    comparative_results = {
        "cache_beats_all_controls": cache_accuracy - best_control_accuracy >= float(comparative["minimum_accuracy_lift"]),
        "history_average_degrades_on_timing_cases": summaries["temporal_state_only"]["timing_required_accuracy"] > summaries["history_averaged_static_interaction"]["timing_required_accuracy"],
        "timing_perturbations_degrade_or_abstain": candidate["timing_perturbation_abstention"] >= 0.9,
        "construction_amortized_by_useful_reuse": bool(candidate["cache_hit_rate"] >= thresholds["cache_hit_rate"]["limit"] and candidate["useful_reuse_rate"] >= thresholds["useful_reuse_rate"]["limit"]),
    }
    threshold_gate = all(item["passed"] for item in threshold_results.values())
    budget_gate = all(item["passed"] for item in budget_results.values())
    comparative_gate = all(comparative_results.values())
    report: Dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "experiment_id": preregistration["experiment_id"],
        "protocol_fingerprint": preregistration["protocol_fingerprint"],
        "fixture_freeze_fingerprint": fixture_manifest["freeze_fingerprint"],
        "decision_freeze": dict(decision_identity),
        "evaluator_key_digest": fixture_manifest["evaluator_key_digest"],
        "decision_count": len(decisions),
        "evaluation_decision_count": len(evaluation_rows),
        "arm_metrics": summaries,
        "threshold_results": threshold_results,
        "budget_results": budget_results,
        "comparative_results": comparative_results,
        "threshold_gate_passed": threshold_gate,
        "budget_gate_passed": budget_gate,
        "comparative_gate_passed": comparative_gate,
        "mechanism_gate_passed": threshold_gate and budget_gate and comparative_gate,
        "promotion_ready": False,
        "production_mutation": False,
        "claim_boundary": "Frozen synthetic temporal controls only; no independent workload, production, ANN-parity, or physical-energy claim.",
    }
    report["report_digest"] = _digest(report)
    return report


__all__ = ["BENCHMARK_SCHEMA", "DECISION_SCHEMA", "evaluate_frozen_decisions", "freeze_decisions"]
