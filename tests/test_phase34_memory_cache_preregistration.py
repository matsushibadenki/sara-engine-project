from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from sara_engine.evaluation.phase34_memory_cache_preregistration import (
    ARMS,
    CASE_FAMILIES,
    EXPERIMENT_ID,
    build_registered_manifest,
    compare_existing_registration,
    validate_preregistration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_draft_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "phase34_memory_checkpoint_cache_draft.py"
    spec = importlib.util.spec_from_file_location("phase34_memory_checkpoint_cache_draft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_draft():
    module = _load_draft_module()
    fixture = PROJECT_ROOT / "data" / "processed" / "benchmark_fixtures" / "phase34_memory_checkpoint_cache_cases.jsonl"
    rows = module.load_fixture(str(fixture))
    return module, rows, module.build_draft(rows, module.environment_descriptor())


def test_fixture_and_preregistration_freeze_bounded_four_arm_experiment():
    _, rows, draft = _valid_draft()
    manifest = build_registered_manifest(draft, managed_path=True)
    validation = validate_preregistration(manifest, managed_path=True)

    assert tuple(row["family"] for row in rows) == CASE_FAMILIES
    assert len(rows) == 16
    assert manifest["experiment_id"] == EXPERIMENT_ID
    assert tuple(manifest["arms"]) == ARMS
    assert manifest["selection"]["selected_k"] == 2
    assert manifest["segmentation"]["parameter_averaging"] is False
    assert manifest["budgets"]["restart_count_per_arm"] == 0
    assert validation["valid"] is True


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda draft: draft["arms"].reverse(), "arms_do_not_match_frozen_phase34"),
        (lambda draft: draft["selection"].update({"selected_k": 3}), "selection_contract_mismatch"),
        (lambda draft: draft["selection"].update({"learned_router": True}), "selection_contract_mismatch"),
        (lambda draft: draft["segmentation"].update({"parameter_averaging": True}), "segmentation_contract_mismatch"),
        (lambda draft: draft["budgets"].update({"restart_count_per_arm": 1}), "budgets_do_not_match_frozen_phase34"),
        (lambda draft: draft["execution_policy"].update({"backpropagation": True}), "execution_policy_does_not_match_sara_boundaries"),
    ],
)
def test_preregistration_rejects_protocol_drift(mutate, error):
    _, _, draft = _valid_draft()
    mutate(draft)
    with pytest.raises(ValueError, match=error):
        build_registered_manifest(draft, managed_path=True)


def test_registration_is_idempotent_and_immutable():
    _, _, draft = _valid_draft()
    manifest = build_registered_manifest(draft, managed_path=True)
    assert compare_existing_registration(manifest, manifest) == (True, "identical_registration_preserved")

    changed = copy.deepcopy(manifest)
    changed["thresholds"]["selection_precision"]["limit"] = 0.8
    assert compare_existing_registration(manifest, changed) == (False, "existing_registration_is_immutable")


def test_fixture_rejects_missing_family_and_durable_mutation():
    module, rows, _ = _valid_draft()
    with pytest.raises(ValueError, match="case_families"):
        module.validate_fixture(rows[:-1])

    mutable = copy.deepcopy(rows)
    mutable[0]["expected"]["durable_mutation_allowed"] = True
    with pytest.raises(ValueError, match="expected_contract"):
        module.validate_fixture(mutable)
