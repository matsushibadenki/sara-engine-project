from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from sara_engine.evaluation.phase33_twinprop_preregistration import (
    ABLATION_ARMS,
    CASE_FAMILIES,
    EXPERIMENT_ID,
    build_registered_manifest,
    compare_existing_registration,
    validate_preregistration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_draft_module():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "eval"
        / "phase33_twinprop_ablation_draft.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase33_twinprop_ablation_draft",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_draft():
    module = _load_draft_module()
    fixture = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_fixtures"
        / "phase33_twinprop_ablation_cases.jsonl"
    )
    rows = module.load_fixture(str(fixture))
    return module, rows, module.build_draft(rows, module.environment_descriptor())


def test_twinprop_fixture_and_preregistration_freeze_separate_followup():
    _, rows, draft = _valid_draft()
    manifest = build_registered_manifest(draft, managed_path=True)
    validation = validate_preregistration(manifest, managed_path=True)

    assert tuple(row["family"] for row in rows) == CASE_FAMILIES
    assert len(rows) == 14
    assert manifest["experiment_id"] == EXPERIMENT_ID
    assert tuple(manifest["ablation_arms"]) == ABLATION_ARMS
    assert manifest["fixed_readout"]["trainable"] is False
    assert manifest["fixed_readout"]["deep_decoder_allowed"] is False
    assert manifest["budgets"]["restart_count_per_arm"] == 0
    assert validation["valid"] is True


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda draft: draft["ablation_arms"].reverse(),
            "ablation_arms_do_not_match_frozen_followup",
        ),
        (
            lambda draft: draft["fixed_readout"].update({"trainable": True}),
            "fixed_readout_contract_mismatch",
        ),
        (
            lambda draft: draft["budgets"].update({"restart_count_per_arm": 1}),
            "restart_count_per_arm_must_equal_zero",
        ),
        (
            lambda draft: draft["resource_accounting"].update(
                {"gradient_selected_contact_locations": True}
            ),
            "resource_accounting_contract_mismatch",
        ),
        (
            lambda draft: draft["execution_policy"].update(
                {"backpropagation": True}
            ),
            "execution_policy_does_not_match_sara_boundaries",
        ),
    ],
)
def test_twinprop_preregistration_rejects_protocol_drift(mutate, error):
    _, _, draft = _valid_draft()
    mutate(draft)
    with pytest.raises(ValueError, match=error):
        build_registered_manifest(draft, managed_path=True)


def test_twinprop_registration_is_idempotent_and_immutable():
    _, _, draft = _valid_draft()
    manifest = build_registered_manifest(draft, managed_path=True)
    assert compare_existing_registration(manifest, manifest) == (
        True,
        "identical_registration_preserved",
    )

    changed = copy.deepcopy(manifest)
    changed["thresholds"]["intact_over_passive_delta"]["limit"] = 0.2
    assert compare_existing_registration(manifest, changed) == (
        False,
        "existing_registration_is_immutable",
    )


def test_twinprop_fixture_rejects_missing_order_and_mutable_expected_state():
    module, rows, _ = _valid_draft()
    missing_order = [row for row in rows if row["interaction_order"] != 4]
    mutable = [dict(row) for row in rows]
    mutable[0] = {
        **mutable[0],
        "expected": {
            **mutable[0]["expected"],
            "durable_mutation_allowed": True,
        },
    }

    with pytest.raises(ValueError, match="case_families|interaction_order"):
        module.validate_fixture(missing_order)
    with pytest.raises(ValueError, match="expected_contract"):
        module.validate_fixture(mutable)
