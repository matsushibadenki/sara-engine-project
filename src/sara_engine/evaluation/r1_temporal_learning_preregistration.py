"""Fail-closed preregistration for the R1 causal temporal-learning experiment."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from typing import Any, Dict, Mapping, Tuple


SCHEMA = "sara-r1-causal-temporal-learning-preregistration-v1"
ARMS: Tuple[str, ...] = (
    "intact_spiking_local",
    "frozen_spiking",
    "shuffled_outcome_feedback",
    "timing_destroyed_spiking",
    "bounded_transition_memory",
    "nonspiking_temporal_state",
)
TASK_FAMILIES: Tuple[str, ...] = (
    "ordered_cue_short_gap",
    "ordered_cue_long_gap",
    "same_multiset_reversed_order",
    "interval_coded_cue",
    "distractor_resistant_cue",
    "controlled_rule_reversal",
    "timing_irrelevant_control",
)
HELDOUT_GENERATOR_FAMILIES: Tuple[str, ...] = (
    "heldout_symbol_permutation",
    "heldout_interval_offset",
)
REQUIRED_BUDGETS: Tuple[str, ...] = (
    "max_events_per_episode",
    "max_units",
    "max_synapses",
    "max_active_neighbors_per_event",
    "max_eligibility_traces",
    "max_trace_age_steps",
    "max_updates_per_event",
    "max_readout_entries",
    "max_replay_events",
    "max_state_bytes",
    "max_event_work_per_episode",
    "max_cpu_ms_per_episode",
    "development_configurations",
    "evaluation_runs_per_hypothesis",
)
REQUIRED_METRICS: Tuple[str, ...] = (
    "primary_temporal_accuracy",
    "timing_irrelevant_accuracy",
    "correction_latency_episodes",
    "old_rule_interference_rate",
    "answered_coverage",
    "answered_error_rate",
    "work_per_correct_prediction",
    "resource_contract_pass_rate",
)


def _digest(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("protocol_fingerprint", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def is_managed_preregistration_path(path: str) -> bool:
    return f"{os.sep}workspace{os.sep}" in os.path.realpath(path)


def validate_preregistration(manifest: Mapping[str, Any], *, managed_path: bool) -> Dict[str, Any]:
    errors = []
    if not managed_path:
        errors.append("preregistration_path_not_managed")
    if manifest.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if manifest.get("registered_before_candidate_implementation") is not True:
        errors.append("candidate_implementation_boundary_not_frozen")
    if tuple(manifest.get("arms", ())) != ARMS:
        errors.append("arms_do_not_match_frozen_protocol")
    if tuple(manifest.get("task_families", ())) != TASK_FAMILIES:
        errors.append("task_families_do_not_match_frozen_protocol")
    if tuple(manifest.get("heldout_generator_families", ())) != HELDOUT_GENERATOR_FAMILIES:
        errors.append("heldout_generators_do_not_match_frozen_protocol")

    seeds = manifest.get("training_seeds", ())
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        errors.append("at_least_five_unique_training_seeds_required")
    if set(seeds) & set(manifest.get("evaluation_seeds", ())):
        errors.append("training_and_evaluation_seeds_overlap")

    budgets = manifest.get("budgets", {})
    missing = [key for key in REQUIRED_BUDGETS if key not in budgets]
    if missing:
        errors.append("missing_budgets:" + ",".join(missing))
    elif any(
        not isinstance(budgets[key], (int, float))
        or budgets[key] < 0
        or (key != "max_replay_events" and budgets[key] == 0)
        for key in REQUIRED_BUDGETS
    ):
        errors.append("budgets_must_be_positive_except_zero_replay")

    metrics = manifest.get("metrics", {})
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        errors.append("missing_metrics:" + ",".join(missing))

    threshold = manifest.get("decision_threshold", {})
    expected_threshold = {
        "minimum_gain_percentage_points": 5.0,
        "paired_ci_lower_bound_above_zero": True,
        "must_beat_frozen": True,
        "must_beat_strongest_nonspiking": True,
        "must_beat_shuffled_feedback": True,
        "timing_destruction_must_remove_temporal_gain": True,
        "timing_irrelevant_control_must_not_show_comparable_loss": True,
        "all_resource_contracts_must_pass": True,
    }
    if any(threshold.get(key) != value for key, value in expected_threshold.items()):
        errors.append("decision_threshold_does_not_match_r1")

    statistics = manifest.get("statistics", {})
    if (
        statistics.get("unit") != "independent_episode"
        or statistics.get("interval") != "paired_percentile_bootstrap_95"
        or statistics.get("bootstrap_resamples") != 10000
        or statistics.get("multiplicity") != "holm_three_primary_comparisons"
        or not isinstance(statistics.get("bootstrap_seed"), int)
    ):
        errors.append("statistical_protocol_incomplete")

    leakage = manifest.get("leakage_policy", {})
    required_leakage = (
        "outcome_hidden_until_prediction_frozen",
        "generator_family_disjoint_split",
        "sequence_identity_disjoint_split",
        "revision_identity_disjoint_split",
        "same_encoder_across_temporal_arms",
        "same_target_exposure_across_arms",
        "same_seeds_across_arms",
        "no_direct_answer_cache",
        "no_teacher_or_external_model",
        "preprocessing_work_counted",
        "abstentions_counted",
    )
    if any(leakage.get(key) is not True for key in required_leakage):
        errors.append("leakage_policy_incomplete")

    execution = manifest.get("execution_policy", {})
    expected_execution = {
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "global_backpropagation": False,
        "external_model": False,
        "production_mutation": False,
        "durable_memory_mutation": False,
        "physical_energy_claim": False,
    }
    if any(execution.get(key) != value for key, value in expected_execution.items()):
        errors.append("execution_policy_does_not_match_r1")

    prereq = manifest.get("prerequisites", {})
    if prereq.get("r0_contracts_passed") is not True or not prereq.get("r0_probe_sha256"):
        errors.append("r0_prerequisite_missing")
    if manifest.get("failure_policy", {}).get("architectural_expansion_after_two_failed_hypotheses") != "stop":
        errors.append("failure_stop_rule_missing")

    fingerprint = manifest.get("protocol_fingerprint")
    if fingerprint is not None and fingerprint != _digest(manifest):
        errors.append("protocol_fingerprint_mismatch")
    return {"valid": not errors, "errors": errors}


def build_registered_manifest(draft: Mapping[str, Any], *, managed_path: bool) -> Dict[str, Any]:
    candidate = deepcopy(dict(draft))
    candidate.pop("protocol_fingerprint", None)
    validation = validate_preregistration(candidate, managed_path=managed_path)
    if not validation["valid"]:
        raise ValueError(";".join(validation["errors"]))
    candidate["protocol_fingerprint"] = _digest(candidate)
    return candidate


def compare_existing_registration(existing: Mapping[str, Any], candidate: Mapping[str, Any]) -> Tuple[bool, str]:
    if not existing:
        return True, "new_registration"
    if dict(existing) == dict(candidate):
        return True, "identical_registration_preserved"
    return False, "existing_registration_is_immutable"


__all__ = [
    "ARMS", "HELDOUT_GENERATOR_FAMILIES", "REQUIRED_BUDGETS", "REQUIRED_METRICS",
    "SCHEMA", "TASK_FAMILIES", "build_registered_manifest",
    "compare_existing_registration", "is_managed_preregistration_path",
    "validate_preregistration",
]
