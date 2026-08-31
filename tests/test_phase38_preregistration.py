from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.phase38_preregistration import build_registered_manifest, compare_existing_registration, validate_preregistration


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "workspace" / "evaluation" / "phase38_structural_delta_preregistration_draft.json"


def _draft():
    return json.loads(DRAFT.read_text(encoding="utf-8"))


def test_phase38_complete_protocol_is_valid_and_fingerprinted():
    manifest = build_registered_manifest(_draft(), managed_path=True)
    assert validate_preregistration(manifest, managed_path=True) == {"valid": True, "errors": []}
    assert len(manifest["protocol_fingerprint"]) == 64
    assert manifest["delta_contract"]["remove_creates_tombstone"] is True


def test_phase38_registration_is_immutable():
    manifest = build_registered_manifest(_draft(), managed_path=True)
    assert compare_existing_registration(manifest, manifest) == (True, "identical_registration_preserved")
    changed = copy.deepcopy(manifest)
    changed["description_cost_table"]["delta_header"] += 1
    assert compare_existing_registration(manifest, changed) == (False, "existing_registration_is_immutable")


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda d: d["arms"].pop(), "arms_do_not_match_frozen_protocol"),
        (lambda d: d["operators"].remove("SPLIT"), "operators_do_not_match_frozen_vocabulary"),
        (lambda d: d["replicate_seeds"].pop(), "at_least_five_unique_seeds_required"),
        (lambda d: d["budgets"].pop("max_chain_depth"), "missing_budgets:max_chain_depth"),
        (lambda d: d["description_cost_table"].pop("decoder_byte"), "missing_description_costs:decoder_byte"),
        (lambda d: d["delta_contract"].update({"inverse_required": False}), "delta_contract_incomplete"),
        (lambda d: d["leakage_policy"].update({"target_aware_base_selection_forbidden": False}), "leakage_policy_incomplete"),
        (lambda d: d["execution_policy"].update({"snapshot_format_mutation": True}), "execution_policy_mismatch"),
    ],
)
def test_phase38_protocol_drift_fails_closed(mutation, error):
    draft = _draft()
    mutation(draft)
    with pytest.raises(ValueError, match=error):
        build_registered_manifest(draft, managed_path=True)


def test_phase38_unmanaged_registration_fails_closed():
    with pytest.raises(ValueError, match="preregistration_path_not_managed"):
        build_registered_manifest(_draft(), managed_path=False)
