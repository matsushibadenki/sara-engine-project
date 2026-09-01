from __future__ import annotations

import copy

import pytest

from sara_engine.evaluation.phase30_fixtures import (
    EVENTS_PER_CASE,
    PARTITIONS,
    REPLICATE_SEEDS,
    build_fixtures,
    validate_fixtures,
)
from sara_engine.evaluation.phase30_preregistration import CASE_FAMILIES


def test_phase30_fixtures_are_complete_deterministic_and_isolated():
    inputs, keys, manifest = build_fixtures()
    repeated = build_fixtures()
    assert (inputs, keys, manifest) == repeated
    assert len(inputs) == len(PARTITIONS) * len(CASE_FAMILIES) * len(REPLICATE_SEEDS)
    assert manifest["event_count"] == len(inputs) * EVENTS_PER_CASE
    assert manifest["source_generator_disjoint"] is True
    assert manifest["evaluator_labels_absent_from_inputs"] is True
    assert {case["partition"] for case in inputs} == set(PARTITIONS)
    assert {case["case_family"] for case in inputs} == set(CASE_FAMILIES)


def test_phase30_evaluator_labels_are_physically_separate():
    inputs, keys, _ = build_fixtures()
    assert all("expected_decision" not in case for case in inputs)
    assert all("timing_required" not in case for case in inputs)
    assert all("expected_decision" in case for case in keys)
    assert {case["case_id"] for case in inputs} == {case["case_id"] for case in keys}


@pytest.mark.parametrize("target", ("input", "key", "manifest"))
def test_phase30_fixture_tampering_fails_closed(target):
    inputs, keys, manifest = build_fixtures()
    inputs, keys, manifest = copy.deepcopy(inputs), copy.deepcopy(keys), copy.deepcopy(manifest)
    if target == "input":
        inputs[0]["events"][0]["phase_bucket"] += 1
    elif target == "key":
        keys[0]["expected_decision"] = "tampered"
    else:
        manifest["case_count"] += 1
    with pytest.raises(ValueError):
        validate_fixtures(inputs, keys, manifest)
