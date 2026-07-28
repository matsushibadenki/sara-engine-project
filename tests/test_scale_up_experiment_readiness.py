from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "scale_up_experiment_readiness.py"
    spec = importlib.util.spec_from_file_location("scale_up_experiment_readiness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_preregistration(module):
    manifest = {
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
        "profiles": [
            "frozen_control",
            "event_memory",
            "structural_feedback_event_memory",
        ],
        "episode_buckets": [1000, 10000],
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
            for metric in module.REQUIRED_METRICS
        },
        "execution_policy": {
            "cpu_only": True,
            "gpu_required": False,
            "network_collection": False,
            "physical_energy_claim": False,
        },
    }
    manifest["protocol_fingerprint"] = module.preregistration_fingerprint(
        manifest
    )
    return manifest


def test_scale_up_readiness_stays_blocked_until_gates_pass():
    module = _load_module()
    report = module.build_readiness(
        {"promotion_allowed": False},
        {"promotion_allowed": False},
        {"promotion_allowed": False},
    )

    assert report["ready_to_execute"] is False
    assert report["plan"]["episode_buckets"] == [1000, 10000]
    assert "next_level_promotion_gate_blocked" in report["blockers"]
    assert "scale_up_preregistration_invalid" in report["blockers"]


def test_scale_up_readiness_can_open_only_when_all_gates_and_manifest_pass():
    module = _load_module()
    manifest = _valid_preregistration(module)
    report = module.build_readiness(
        {"promotion_allowed": True},
        {"promotion_allowed": True},
        {"promotion_allowed": True},
        manifest,
        preregistration_path_managed=True,
    )

    assert report["ready_to_execute"] is True
    assert report["preregistration"]["valid"] is True
    assert report["plan"]["execution_policy"]["external_device_required"] is False


def test_scale_up_readiness_rejects_stale_or_unequal_preregistration():
    module = _load_module()
    manifest = _valid_preregistration(module)
    manifest["budgets"]["events_per_episode"] = 257
    report = module.build_readiness(
        {"promotion_allowed": True},
        {"promotion_allowed": True},
        {"promotion_allowed": True},
        manifest,
        preregistration_path_managed=True,
    )

    assert report["ready_to_execute"] is False
    assert "event_budget_does_not_match_frozen_protocol" in (
        report["preregistration"]["errors"]
    )
    assert "protocol_fingerprint_mismatch" in report["preregistration"]["errors"]


def test_scale_up_readiness_rejects_unmanaged_preregistration_path():
    module = _load_module()
    manifest = _valid_preregistration(module)
    report = module.build_readiness(
        {"promotion_allowed": True},
        {"promotion_allowed": True},
        {"promotion_allowed": True},
        manifest,
        preregistration_path_managed=False,
    )

    assert report["ready_to_execute"] is False
    assert "preregistration_path_not_managed" in (
        report["preregistration"]["errors"]
    )


def test_scale_up_readiness_requires_explicit_multimodal_gate():
    module = _load_module()
    manifest = _valid_preregistration(module)
    report = module.build_readiness(
        {"promotion_allowed": True},
        {"promotion_allowed": True},
        None,
        manifest,
        preregistration_path_managed=True,
    )

    assert report["ready_to_execute"] is False
    assert "independent_multimodal_coverage_incomplete" in report["blockers"]


def test_scale_up_preregistration_validation_rejects_malformed_shapes():
    module = _load_module()
    manifest = _valid_preregistration(module)
    manifest["profiles"] = None
    manifest["episode_buckets"] = 1000
    manifest["thresholds"]["latency"] = "fast"
    manifest["protocol_fingerprint"] = "0" * 64

    result = module.validate_preregistration(
        manifest,
        managed_path=True,
    )

    assert result["valid"] is False
    assert "profiles_do_not_match_frozen_protocol" in result["errors"]
    assert "episode_buckets_do_not_match_frozen_protocol" in result["errors"]
    assert "invalid_threshold_spec:latency" in result["errors"]
