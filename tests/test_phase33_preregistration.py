from __future__ import annotations

import copy

import pytest

from sara_engine.evaluation.phase33_preregistration import (
    MECHANISM_ARMS,
    REQUIRED_BUDGETS,
    REQUIRED_CASE_FAMILIES,
    REQUIRED_METRICS,
    SCHEMA,
    SIMPLIFICATION_LEVELS,
    build_registered_manifest,
    compare_existing_registration,
    validate_preregistration,
)


def _valid_draft():
    return {
        "schema": SCHEMA,
        "experiment_id": "phase33-structured-edge-v1",
        "registered_before_execution": True,
        "fixture_fingerprint": "a" * 64,
        "environment_fingerprint": "b" * 64,
        "mechanism_arms": list(MECHANISM_ARMS),
        "simplification_levels": [
            dict(level) for level in SIMPLIFICATION_LEVELS
        ],
        "case_families": list(REQUIRED_CASE_FAMILIES),
        "replicates_per_condition": 5,
        "replicate_seeds": [101, 211, 307, 401, 503],
        "budgets": {
            "source_events_per_case": 128,
            "max_total_state_bytes": 4096,
            "max_local_interactions_per_case": 256,
            "max_latency_ms": 50,
            "max_outer_nodes": 64,
            "max_outer_routes": 128,
            "max_contacts_per_relation": 4,
            "max_branch_slots_per_relation": 4,
            "max_internal_interactions_per_relation": 8,
            "max_contact_rewrites_per_event": 2,
        },
        "resource_accounting": {
            "equal_source_events_across_arms": True,
            "same_replicate_seeds_across_arms": True,
            "contacts_count_toward_total_state": True,
            "internal_interactions_count_toward_total_state": True,
            "internal_interactions_count_toward_event_cost": True,
            "same_latency_ceiling_across_arms": True,
            "simplification_may_not_increase_total_budget": True,
        },
        "thresholds": {
            metric: {"direction": "minimum", "limit": 0.0}
            for metric in REQUIRED_METRICS
        },
        "execution_policy": {
            "cpu_only": True,
            "gpu_required": False,
            "matrix_calculation": False,
            "backpropagation": False,
            "default_off": True,
            "production_mutation": False,
            "physical_energy_claim": False,
            "independent_evidence_required": True,
        },
    }


def test_phase33_preregistration_accepts_frozen_complete_protocol():
    manifest = build_registered_manifest(_valid_draft(), managed_path=True)
    validation = validate_preregistration(manifest, managed_path=True)

    assert validation["valid"] is True
    assert validation["errors"] == []
    assert manifest["protocol_fingerprint"]


def test_phase33_preregistration_is_idempotent_and_immutable():
    manifest = build_registered_manifest(_valid_draft(), managed_path=True)
    assert compare_existing_registration(manifest, manifest) == (
        True,
        "identical_registration_preserved",
    )

    changed = copy.deepcopy(manifest)
    changed["budgets"]["max_contacts_per_relation"] = 5
    assert compare_existing_registration(manifest, changed) == (
        False,
        "existing_registration_is_immutable",
    )


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda draft: draft["mechanism_arms"].reverse(),
            "mechanism_arms_do_not_match_frozen_protocol",
        ),
        (
            lambda draft: draft["simplification_levels"].pop(),
            "simplification_levels_do_not_match_frozen_protocol",
        ),
        (
            lambda draft: draft["case_families"].remove("random_cluster"),
            "case_families_do_not_match_frozen_protocol",
        ),
        (
            lambda draft: draft["budgets"].pop(REQUIRED_BUDGETS[0]),
            "missing_budgets:source_events_per_case",
        ),
        (
            lambda draft: draft["resource_accounting"].update(
                {"contacts_count_toward_total_state": False}
            ),
            "resource_accounting_does_not_prevent_hidden_complexity",
        ),
        (
            lambda draft: draft["execution_policy"].update(
                {"backpropagation": True}
            ),
            "execution_policy_does_not_match_phase33_boundaries",
        ),
    ],
)
def test_phase33_preregistration_rejects_protocol_drift(mutate, error):
    draft = _valid_draft()
    mutate(draft)

    with pytest.raises(ValueError, match=error):
        build_registered_manifest(draft, managed_path=True)


def test_phase33_preregistration_rejects_unmanaged_output_boundary():
    with pytest.raises(ValueError, match="preregistration_path_not_managed"):
        build_registered_manifest(_valid_draft(), managed_path=False)


def test_phase33_preregistration_reports_non_string_budget_and_threshold_names():
    draft = _valid_draft()
    draft["budgets"][1] = 1
    draft["thresholds"][2] = {"direction": "minimum", "limit": 0.0}

    with pytest.raises(ValueError) as exc_info:
        build_registered_manifest(draft, managed_path=True)

    message = str(exc_info.value)
    assert "budget_names_must_be_strings" in message
    assert "threshold_names_must_be_strings" in message
