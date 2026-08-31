from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.phase37_preregistration import build_registered_manifest, compare_existing_registration, validate_preregistration


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "workspace" / "evaluation" / "phase37_structural_invariant_preregistration_draft.json"


def _draft():
    return json.loads(DRAFT.read_text(encoding="utf-8"))


def test_phase37_complete_protocol_is_valid_and_fingerprinted():
    manifest = build_registered_manifest(_draft(), managed_path=True)
    assert validate_preregistration(manifest, managed_path=True) == {"valid": True, "errors": []}
    assert len(manifest["protocol_fingerprint"]) == 64
    assert manifest["canonical_identity"]["node_identity_in_fingerprint"] is False


def test_phase37_registration_is_immutable():
    manifest = build_registered_manifest(_draft(), managed_path=True)
    assert compare_existing_registration(manifest, manifest) == (True, "identical_registration_preserved")
    changed = copy.deepcopy(manifest)
    changed["budgets"]["max_patterns"] += 1
    assert compare_existing_registration(manifest, changed) == (False, "existing_registration_is_immutable")


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda d: d["mechanism_arms"].pop(), "mechanism_arms_do_not_match_frozen_protocol"),
        (lambda d: d["ablation_shuffles"].remove("topology"), "ablation_shuffles_do_not_match_frozen_protocol"),
        (lambda d: d["replicate_seeds"].pop(), "at_least_five_unique_seeds_required"),
        (lambda d: d["budgets"].pop("max_patterns"), "missing_budgets:max_patterns"),
        (lambda d: d["canonical_identity"].update({"node_identity_in_fingerprint": True}), "canonical_identity_allows_label_leakage"),
        (lambda d: d["execution_policy"].update({"production_mutation": True}), "execution_policy_does_not_match_phase37_boundaries"),
    ],
)
def test_phase37_protocol_drift_fails_closed(mutation, error):
    draft = _draft()
    mutation(draft)
    with pytest.raises(ValueError, match=error):
        build_registered_manifest(draft, managed_path=True)


def test_phase37_unmanaged_registration_fails_closed():
    with pytest.raises(ValueError, match="preregistration_path_not_managed"):
        build_registered_manifest(_draft(), managed_path=False)
