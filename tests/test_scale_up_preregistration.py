from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    eval_path = PROJECT_ROOT / "scripts" / "eval"
    readiness_path = eval_path / "scale_up_experiment_readiness.py"
    readiness_spec = importlib.util.spec_from_file_location(
        "scale_up_experiment_readiness", readiness_path
    )
    assert readiness_spec is not None and readiness_spec.loader is not None
    readiness = importlib.util.module_from_spec(readiness_spec)
    readiness_spec.loader.exec_module(readiness)

    registration_path = eval_path / "scale_up_preregistration.py"
    registration_spec = importlib.util.spec_from_file_location(
        "scale_up_preregistration", registration_path
    )
    assert (
        registration_spec is not None
        and registration_spec.loader is not None
    )
    registration = importlib.util.module_from_spec(registration_spec)
    registration_spec.loader.exec_module(registration)
    return readiness, registration


def _valid_draft(readiness):
    return {
        "schema": "sara-scale-up-preregistration-v1",
        "registered_before_execution": True,
        "source_fingerprints": {
            "domain-a": "a" * 64,
            "domain-b": "b" * 64,
            "domain-c": "c" * 64,
            "domain-d": "d" * 64,
        },
        "fixture_fingerprint": "e" * 64,
        "environment_fingerprint": "f" * 64,
        "profiles": list(readiness.REQUIRED_PROFILES),
        "episode_buckets": list(readiness.REQUIRED_EPISODE_BUCKETS),
        "replicates_per_condition": 5,
        "replicate_seeds": [101, 211, 307, 401, 503],
        "data_policy": {
            "same_sources_for_all_profiles": True,
            "same_fixture_for_all_profiles": True,
            "fixed_episode_order_per_replicate": True,
        },
        "budgets": {
            "same_for_all_profiles": True,
            "state_entries": 128,
            "events_per_episode": 256,
        },
        "thresholds": {
            metric: {"direction": "minimum", "limit": 0.0}
            for metric in readiness.REQUIRED_METRICS
        },
        "execution_policy": {
            "cpu_only": True,
            "gpu_required": False,
            "network_collection": False,
            "physical_energy_claim": False,
        },
    }


def test_scale_up_registration_adds_valid_protocol_fingerprint():
    readiness, registration = _load_module()
    manifest = registration.build_registered_manifest(
        _valid_draft(readiness),
        managed_path=True,
    )

    assert readiness.validate_preregistration(
        manifest, managed_path=True
    )["valid"] is True


def test_scale_up_registration_is_idempotent_but_immutable():
    readiness, registration = _load_module()
    manifest = registration.build_registered_manifest(
        _valid_draft(readiness),
        managed_path=True,
    )

    assert registration.compare_existing_registration(
        manifest, manifest
    ) == (True, "identical_registration_preserved")
    changed = dict(manifest)
    changed["replicates_per_condition"] = 6
    assert registration.compare_existing_registration(
        manifest, changed
    ) == (False, "existing_registration_is_immutable")


def test_scale_up_registration_rejects_incomplete_draft():
    readiness, registration = _load_module()
    draft = _valid_draft(readiness)
    del draft["thresholds"]["latency"]

    with pytest.raises(ValueError, match="missing_thresholds:latency"):
        registration.build_registered_manifest(draft, managed_path=True)
