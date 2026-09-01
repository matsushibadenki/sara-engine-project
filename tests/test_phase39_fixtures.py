from __future__ import annotations

import copy

import pytest

from sara_engine.evaluation.phase39_fixtures import EVENTS_PER_CASE, FORBIDDEN_INPUT_FIELDS, build_fixtures, validate_fixtures
from sara_engine.evaluation.phase39_preregistration import CASE_FAMILIES


def test_phase39_fixtures_are_complete_deterministic_and_isolated():
    inputs, keys, manifest = build_fixtures()
    assert (inputs, keys, manifest) == build_fixtures()
    assert len(inputs) == 2 * len(CASE_FAMILIES) * 5 == 210
    assert manifest["event_count"] == 210 * EVENTS_PER_CASE == 53760
    assert manifest["source_generator_disjoint"] is True
    assert manifest["evaluator_fields_absent_from_inputs"] is True
    assert {row["case_family"] for row in keys} == set(CASE_FAMILIES)
    assert {row["partition"] for row in keys} == {"train", "evaluation"}


def test_phase39_candidate_inputs_exclude_every_registered_evaluator_field():
    inputs, keys, _ = build_fixtures()
    encoded = str(inputs)
    assert all(field not in encoded for field in FORBIDDEN_INPUT_FIELDS)
    assert all("hidden_factor_ids" in row and "expected_outcome" in row for row in keys)
    assert {row["case_id"] for row in inputs} == {row["case_id"] for row in keys}


@pytest.mark.parametrize("target", ("input", "key", "manifest", "leak"))
def test_phase39_fixture_tampering_and_leakage_fail_closed(target):
    inputs, keys, manifest = build_fixtures()
    inputs, keys, manifest = copy.deepcopy(inputs), copy.deepcopy(keys), copy.deepcopy(manifest)
    if target == "input":
        inputs[0]["events"][0]["phase"] += 1
    elif target == "key":
        keys[0]["expected_outcome"] = "tampered"
    elif target == "manifest":
        manifest["event_count"] += 1
    else:
        inputs[0]["hidden_factor_id"] = "leak"
    with pytest.raises(ValueError):
        validate_fixtures(inputs, keys, manifest)
