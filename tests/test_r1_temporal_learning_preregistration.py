from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.r1_temporal_learning_preregistration import (
    build_registered_manifest,
    compare_existing_registration,
    validate_preregistration,
)


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "workspace" / "evaluation" / "r1_temporal_learning_preregistration_draft.json"


def _draft():
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    draft["prerequisites"]["r0_probe_sha256"] = "a" * 64
    return draft


def test_r1_complete_protocol_is_valid_and_fingerprinted():
    manifest = build_registered_manifest(_draft(), managed_path=True)
    assert validate_preregistration(manifest, managed_path=True) == {"valid": True, "errors": []}
    assert len(manifest["protocol_fingerprint"]) == 64
    assert manifest["budgets"]["max_replay_events"] == 0


def test_r1_registration_is_immutable():
    manifest = build_registered_manifest(_draft(), managed_path=True)
    assert compare_existing_registration(manifest, manifest) == (True, "identical_registration_preserved")
    changed = copy.deepcopy(manifest)
    changed["budgets"]["max_units"] += 1
    assert compare_existing_registration(manifest, changed) == (False, "existing_registration_is_immutable")


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda d: d["arms"].pop(), "arms_do_not_match_frozen_protocol"),
        (lambda d: d["task_families"].pop(), "task_families_do_not_match_frozen_protocol"),
        (lambda d: d["training_seeds"].pop(), "at_least_five_unique_training_seeds_required"),
        (lambda d: d["evaluation_seeds"].append(101), "training_and_evaluation_seeds_overlap"),
        (lambda d: d["budgets"].pop("max_state_bytes"), "missing_budgets:max_state_bytes"),
        (lambda d: d["decision_threshold"].update({"minimum_gain_percentage_points": 4.0}), "decision_threshold_does_not_match_r1"),
        (lambda d: d["leakage_policy"].update({"outcome_hidden_until_prediction_frozen": False}), "leakage_policy_incomplete"),
        (lambda d: d["execution_policy"].update({"matrix_calculation": True}), "execution_policy_does_not_match_r1"),
    ],
)
def test_r1_protocol_drift_fails_closed(mutation, error):
    draft = _draft()
    mutation(draft)
    with pytest.raises(ValueError, match=error):
        build_registered_manifest(draft, managed_path=True)


def test_r1_unmanaged_registration_fails_closed():
    with pytest.raises(ValueError, match="preregistration_path_not_managed"):
        build_registered_manifest(_draft(), managed_path=False)
