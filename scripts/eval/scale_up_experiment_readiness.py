#!/usr/bin/env python3
"""Prepare a larger internal experiment without running it before evidence gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import (  # noqa: E402
    WORKSPACE_DIR,
    ensure_parent_directory,
    workspace_path,
)

DEFAULT_PROMOTION_GATE = workspace_path("evaluation", "next_level_promotion_gate.json")
DEFAULT_EXTERNAL_GATE = workspace_path("evaluation", "continual_horizon_external_gate.json")
DEFAULT_MULTIMODAL_GATE = workspace_path(
    "evaluation", "phase23_external_multimodal_gate.json"
)
DEFAULT_PREREGISTRATION = workspace_path(
    "evaluation", "scale_up_preregistration.json"
)
DEFAULT_OUTPUT = workspace_path("evaluation", "scale_up_experiment_readiness.json")
REQUIRED_PROFILES = (
    "frozen_control",
    "event_memory",
    "structural_feedback_event_memory",
)
REQUIRED_EPISODE_BUCKETS = (1000, 10000)
REQUIRED_METRICS = (
    "revision_uptake_latency",
    "retained_useful_recall",
    "catastrophic_interference",
    "abstention_integrity",
    "state_growth",
    "event_cost",
    "latency",
    "provenance_completeness",
)
ALLOWED_THRESHOLD_DIRECTIONS = frozenset(
    {"maximize", "minimize", "minimum", "maximum"}
)
SHA256_LENGTH = 64


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _is_managed_preregistration_path(path: str) -> bool:
    if not path:
        return False
    resolved = os.path.realpath(os.path.abspath(path))
    workspace_root = os.path.realpath(WORKSPACE_DIR)
    try:
        return os.path.commonpath([resolved, workspace_root]) == workspace_root
    except ValueError:
        return False


def preregistration_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = {
        str(key): value
        for key, value in manifest.items()
        if key != "protocol_fingerprint"
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_preregistration(
    manifest: Mapping[str, Any],
    *,
    managed_path: bool,
) -> Dict[str, Any]:
    errors: List[str] = []
    if not manifest:
        errors.append("preregistration_missing")
        return {
            "valid": False,
            "managed_path": managed_path,
            "computed_fingerprint": None,
            "declared_fingerprint": None,
            "protocol_summary": {},
            "errors": errors,
        }
    if manifest.get("schema") != "sara-scale-up-preregistration-v1":
        errors.append("unsupported_preregistration_schema")
    if manifest.get("registered_before_execution") is not True:
        errors.append("not_registered_before_execution")
    if not managed_path:
        errors.append("preregistration_path_not_managed")

    source_fingerprints = manifest.get("source_fingerprints")
    if not isinstance(source_fingerprints, Mapping):
        errors.append("source_fingerprints_must_be_mapping")
    else:
        domains = [
            key
            for key in source_fingerprints
            if isinstance(key, str) and key
        ]
        fingerprints = list(source_fingerprints.values())
        if len(domains) < 4 or len(domains) != len(source_fingerprints):
            errors.append("at_least_four_named_domains_required")
        if not all(_is_sha256(value) for value in fingerprints):
            errors.append("invalid_source_fingerprint")
        if all(isinstance(value, str) for value in fingerprints) and (
            len(set(fingerprints)) != len(fingerprints)
        ):
            errors.append("source_fingerprints_must_be_unique")

    for field in ("fixture_fingerprint", "environment_fingerprint"):
        if not _is_sha256(manifest.get(field)):
            errors.append(f"invalid_{field}")

    profiles = manifest.get("profiles")
    if (
        not isinstance(profiles, (list, tuple))
        or tuple(profiles) != REQUIRED_PROFILES
    ):
        errors.append("profiles_do_not_match_frozen_protocol")
    episode_buckets = manifest.get("episode_buckets")
    if (
        not isinstance(episode_buckets, (list, tuple))
        or tuple(episode_buckets) != REQUIRED_EPISODE_BUCKETS
    ):
        errors.append("episode_buckets_do_not_match_frozen_protocol")
    if manifest.get("replicates_per_condition") != 5:
        errors.append("replicates_per_condition_must_equal_five")
    replicate_seeds = manifest.get("replicate_seeds")
    if (
        not isinstance(replicate_seeds, list)
        or len(replicate_seeds) != 5
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            for seed in replicate_seeds
        )
        or len(set(replicate_seeds)) != 5
    ):
        errors.append("five_unique_non_negative_replicate_seeds_required")

    data_policy = manifest.get("data_policy")
    if not isinstance(data_policy, Mapping):
        errors.append("data_policy_must_be_mapping")
    else:
        required_data_policy = {
            "same_sources_for_all_profiles": True,
            "same_fixture_for_all_profiles": True,
            "fixed_episode_order_per_replicate": True,
        }
        if any(
            data_policy.get(key) is not value
            for key, value in required_data_policy.items()
        ):
            errors.append("data_policy_does_not_preserve_equal_comparison")

    budgets = manifest.get("budgets")
    if not isinstance(budgets, Mapping):
        errors.append("budgets_must_be_mapping")
    else:
        if budgets.get("same_for_all_profiles") is not True:
            errors.append("budgets_must_be_equal_across_profiles")
        if budgets.get("state_entries") != 128:
            errors.append("state_budget_does_not_match_frozen_protocol")
        if budgets.get("events_per_episode") != 256:
            errors.append("event_budget_does_not_match_frozen_protocol")

    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, Mapping):
        errors.append("thresholds_must_be_mapping")
    else:
        threshold_keys = {
            key for key in thresholds if isinstance(key, str)
        }
        if len(threshold_keys) != len(thresholds):
            errors.append("threshold_names_must_be_strings")
        missing_metrics = sorted(
            set(REQUIRED_METRICS).difference(threshold_keys)
        )
        extra_metrics = sorted(
            threshold_keys.difference(REQUIRED_METRICS)
        )
        if missing_metrics:
            errors.append(
                "missing_thresholds:" + ",".join(missing_metrics)
            )
        if extra_metrics:
            errors.append(
                "unknown_thresholds:" + ",".join(extra_metrics)
            )
        for metric in REQUIRED_METRICS:
            threshold = thresholds.get(metric)
            if not isinstance(threshold, Mapping):
                if metric in thresholds:
                    errors.append(f"invalid_threshold_spec:{metric}")
                continue
            direction = threshold.get("direction")
            limit = threshold.get("limit")
            if direction not in ALLOWED_THRESHOLD_DIRECTIONS:
                errors.append(f"invalid_threshold_direction:{metric}")
            if (
                isinstance(limit, bool)
                or not isinstance(limit, (int, float))
                or not math.isfinite(float(limit))
            ):
                errors.append(f"invalid_threshold_limit:{metric}")

    policy = manifest.get("execution_policy")
    if not isinstance(policy, Mapping):
        errors.append("execution_policy_must_be_mapping")
    else:
        required_policy = {
            "cpu_only": True,
            "gpu_required": False,
            "network_collection": False,
            "physical_energy_claim": False,
        }
        if any(policy.get(key) is not value for key, value in required_policy.items()):
            errors.append("execution_policy_does_not_match_sparse_cpu_protocol")

    try:
        computed_fingerprint = preregistration_fingerprint(manifest)
    except (TypeError, ValueError):
        computed_fingerprint = None
        errors.append("preregistration_is_not_canonical_json")
    declared_fingerprint = manifest.get("protocol_fingerprint")
    if (
        computed_fingerprint is None
        or declared_fingerprint != computed_fingerprint
    ):
        errors.append("protocol_fingerprint_mismatch")

    return {
        "valid": not errors,
        "managed_path": managed_path,
        "computed_fingerprint": computed_fingerprint,
        "declared_fingerprint": declared_fingerprint,
        "protocol_summary": {
            "domains": sorted(
                str(key)
                for key in source_fingerprints
            )
            if isinstance(source_fingerprints, Mapping)
            else [],
            "profiles": list(profiles)
            if isinstance(profiles, (list, tuple))
            else [],
            "episode_buckets": list(episode_buckets)
            if isinstance(episode_buckets, (list, tuple))
            else [],
            "replicate_seeds": list(replicate_seeds)
            if isinstance(replicate_seeds, list)
            else [],
            "threshold_count": len(thresholds)
            if isinstance(thresholds, Mapping)
            else 0,
            "budgets": dict(budgets)
            if isinstance(budgets, Mapping)
            else {},
            "equal_data_policy": dict(data_policy)
            if isinstance(data_policy, Mapping)
            else {},
        },
        "errors": errors,
    }


def build_readiness(
    promotion_gate: Mapping[str, Any],
    external_gate: Mapping[str, Any],
    multimodal_gate: Mapping[str, Any] | None = None,
    preregistration: Mapping[str, Any] | None = None,
    *,
    preregistration_path_managed: bool = False,
) -> Dict[str, Any]:
    blockers = []
    if not bool(promotion_gate.get("promotion_allowed", False)):
        blockers.append("next_level_promotion_gate_blocked")
    if not bool(external_gate.get("promotion_allowed", False)):
        blockers.append("independent_horizon_coverage_incomplete")
    if multimodal_gate is None or not bool(
        multimodal_gate.get("promotion_allowed", False)
    ):
        blockers.append("independent_multimodal_coverage_incomplete")
    preregistration_status = validate_preregistration(
        preregistration or {},
        managed_path=preregistration_path_managed,
    )
    if not preregistration_status["valid"]:
        blockers.append("scale_up_preregistration_invalid")
    plan = {
        "profiles": list(REQUIRED_PROFILES),
        "episode_buckets": list(REQUIRED_EPISODE_BUCKETS),
        "domains": 4,
        "replicates_per_condition": 5,
        "fixed_unique_replicate_seeds": 5,
        "equal_data_across_profiles": True,
        "equal_state_budget": 128,
        "equal_event_budget_per_episode": 256,
        "metrics": list(REQUIRED_METRICS),
        "execution_policy": {
            "cpu_only": True,
            "gpu_required": False,
            "external_device_required": False,
            "physical_energy_claim": False,
            "network_collection": False,
        },
    }
    return {
        "schema": "sara-scale-up-experiment-readiness-v2",
        "ready_to_execute": not blockers,
        "observed_only": True,
        "blockers": blockers,
        "plan": plan,
        "preregistration": preregistration_status,
        "required_before_execution": [
            "complete independent 10/30/100 horizon coverage",
            "complete independent multimodal decision and provenance coverage",
            "complete human promotion review",
            "freeze fixture, source, and environment fingerprints",
            "record pre-registered thresholds before execution",
        ],
        "policy": "planning_only; no large run or promotion is performed by this command",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-gate", default=DEFAULT_PROMOTION_GATE)
    parser.add_argument("--external-gate", default=DEFAULT_EXTERNAL_GATE)
    parser.add_argument("--multimodal-gate", default=DEFAULT_MULTIMODAL_GATE)
    parser.add_argument(
        "--preregistration-path", default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_readiness(
        _read_json(args.promotion_gate),
        _read_json(args.external_gate),
        _read_json(args.multimodal_gate),
        _read_json(args.preregistration_path),
        preregistration_path_managed=_is_managed_preregistration_path(
            args.preregistration_path
        ),
    )
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
