"""Fail-closed Phase 30 temporal effective-interaction preregistration."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from typing import Any, Dict, Mapping, Tuple


SCHEMA = "sara-phase30-temporal-effective-interaction-preregistration-v1"
ARMS: Tuple[str, ...] = ("fixed_sparse_snn", "history_averaged_static_interaction", "temporal_state_only", "temporal_state_bounded_effective_interaction")
CASE_FAMILIES: Tuple[str, ...] = ("firing_order_reversal", "equal_count_different_interval", "delayed_response", "phase_synchrony_discrimination", "irregular_event_gaps", "context_revision", "shuffled_time", "phase_shifted", "duplicate_event", "stale_cache", "contradiction", "unseen_context", "no_reuse")
REQUIRED_BUDGETS: Tuple[str, ...] = ("source_events_per_case", "max_active_edges", "max_recent_events_per_edge", "max_cached_interactions", "max_state_bytes", "max_cache_bytes", "max_event_cost", "max_cpu_latency_ms", "timestamp_resolution_us", "max_tuning_attempts")
REQUIRED_METRICS: Tuple[str, ...] = ("timing_required_accuracy", "calibration", "justified_abstention", "timing_sensitivity_delta", "revision_recovery_events", "stale_cache_harm", "cache_hit_rate", "useful_reuse_rate", "construction_event_cost", "invalidation_event_cost", "deterministic_replay")


def _digest(value: Mapping[str, Any]) -> str:
    payload=deepcopy(dict(value)); payload.pop("protocol_fingerprint",None)
    return sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def is_managed_preregistration_path(path:str)->bool: return f"{os.sep}workspace{os.sep}" in os.path.realpath(path)


def validate_preregistration(manifest:Mapping[str,Any],*,managed_path:bool)->Dict[str,Any]:
    errors=[]
    if not managed_path: errors.append("preregistration_path_not_managed")
    if manifest.get("schema")!=SCHEMA: errors.append("schema_mismatch")
    if manifest.get("registered_before_candidate_implementation") is not True: errors.append("implementation_boundary_not_frozen")
    if tuple(manifest.get("arms",()))!=ARMS: errors.append("arms_do_not_match_frozen_protocol")
    if tuple(manifest.get("case_families",()))!=CASE_FAMILIES: errors.append("case_families_do_not_match_frozen_protocol")
    seeds=manifest.get("replicate_seeds",())
    if len(seeds)<5 or len(set(seeds))!=len(seeds): errors.append("at_least_five_unique_seeds_required")
    budgets=manifest.get("budgets",{}); missing=[key for key in REQUIRED_BUDGETS if key not in budgets]
    if missing: errors.append("missing_budgets:"+",".join(missing))
    elif any(not isinstance(budgets[key],(int,float)) or budgets[key]<=0 for key in REQUIRED_BUDGETS): errors.append("budgets_must_be_positive")
    thresholds=manifest.get("thresholds",{}); missing=[key for key in REQUIRED_METRICS if key not in thresholds]
    if missing: errors.append("missing_thresholds:"+",".join(missing))
    state=manifest.get("temporal_state_contract",{})
    required=("timestamp","order","interval","delay","phase_bucket","excitation","fatigue","expiry","provenance_reference")
    if any(field not in state.get("fields",()) for field in required): errors.append("temporal_state_contract_incomplete")
    scalar=manifest.get("finite_scalar_ranges",{})
    if not scalar or any(not isinstance(value,list) or len(value)!=2 or value[0]>=value[1] for value in scalar.values()): errors.append("finite_scalar_ranges_invalid")
    leakage=manifest.get("leakage_policy",{})
    required_leak=("split_by_source_and_temporal_generator","same_generator_seed_same_partition","answer_hidden_until_decision_frozen","evaluator_labels_absent_from_candidate_trace","equal_source_events_across_arms","equal_active_edge_state_event_latency_budgets")
    if any(leakage.get(key) is not True for key in required_leak): errors.append("leakage_or_fairness_policy_incomplete")
    invalidation=manifest.get("invalidation_contract",{})
    if any(invalidation.get(key) is not True for key in ("context_revision","contradiction","expiry","distribution_shift","unstable_oscillation")): errors.append("invalidation_contract_incomplete")
    execution=manifest.get("execution_policy",{}); expected={"cpu_only":True,"gpu_required":False,"matrix_calculation":False,"backpropagation":False,"ann_runtime_dependency":False,"default_off":True,"production_mutation":False,"durable_risa_mutation":False,"physical_energy_claim":False,"human_approval_required_for_integration":True}
    if any(execution.get(key)!=value for key,value in expected.items()): errors.append("execution_policy_mismatch")
    fingerprint=manifest.get("protocol_fingerprint")
    if fingerprint is not None and fingerprint!=_digest(manifest): errors.append("protocol_fingerprint_mismatch")
    return {"valid":not errors,"errors":errors}


def build_registered_manifest(draft:Mapping[str,Any],*,managed_path:bool)->Dict[str,Any]:
    candidate=deepcopy(dict(draft)); candidate.pop("protocol_fingerprint",None); validation=validate_preregistration(candidate,managed_path=managed_path)
    if not validation["valid"]: raise ValueError(";".join(validation["errors"]))
    candidate["protocol_fingerprint"]=_digest(candidate); return candidate


def compare_existing_registration(existing:Mapping[str,Any],candidate:Mapping[str,Any])->Tuple[bool,str]:
    if not existing:return True,"new_registration"
    if dict(existing)==dict(candidate):return True,"identical_registration_preserved"
    return False,"existing_registration_is_immutable"


__all__=["ARMS","CASE_FAMILIES","REQUIRED_BUDGETS","REQUIRED_METRICS","SCHEMA","build_registered_manifest","compare_existing_registration","is_managed_preregistration_path","validate_preregistration"]
