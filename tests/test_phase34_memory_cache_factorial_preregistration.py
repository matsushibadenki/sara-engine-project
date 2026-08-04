from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from sara_engine.evaluation.phase34_factorial_preregistration import (
    ARMS,
    CASE_FAMILIES,
    EXPERIMENT_ID,
    REPLICATE_SEEDS,
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
        / "phase34_memory_cache_factorial_draft.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase34_memory_cache_factorial_draft", path
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
        / "phase34_memory_cache_factorial_cases.jsonl"
    )
    rows = module.load_fixture(str(fixture))
    return module, rows, module.build_draft(rows, module.environment_descriptor())


def test_factorial_freezes_retention_and_selection_as_independent_factors():
    _, rows, draft = _valid_draft()
    manifest = build_registered_manifest(draft, managed_path=True)
    validation = validate_preregistration(manifest, managed_path=True)

    assert tuple(row["family"] for row in rows) == CASE_FAMILIES
    assert len(rows) == 12
    assert len(ARMS) == 5
    assert manifest["experiment_id"] == EXPERIMENT_ID
    assert tuple(manifest["replicate_seeds"]) == REPLICATE_SEEDS
    assert manifest["factorial_design"]["same_retained_set_within_retention_pair"] is True
    assert manifest["factorial_design"]["query_visible_during_retention"] is False
    assert manifest["resource_accounting"]["retained_set_digest_must_match_within_pair"] is True
    assert validation["valid"] is True


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda draft: draft["factorial_design"].update(
                {"query_visible_during_retention": True}
            ),
            "frozen_factorial_mismatch:factorial_design",
        ),
        (
            lambda draft: draft["resource_accounting"].update(
                {"query_aware_admission_allowed": True}
            ),
            "frozen_factorial_mismatch:resource_accounting",
        ),
        (
            lambda draft: draft["replicate_seeds"].pop(),
            "frozen_factorial_mismatch:replicate_seeds",
        ),
        (
            lambda draft: draft["execution_policy"].update(
                {"backpropagation": True}
            ),
            "frozen_factorial_mismatch:execution_policy",
        ),
    ],
)
def test_factorial_rejects_protocol_drift(mutate, error):
    _, _, draft = _valid_draft()
    mutate(draft)
    with pytest.raises(ValueError, match=error):
        build_registered_manifest(draft, managed_path=True)


def test_factorial_registration_is_idempotent_and_immutable():
    _, _, draft = _valid_draft()
    manifest = build_registered_manifest(draft, managed_path=True)
    assert compare_existing_registration(manifest, manifest) == (
        True,
        "identical_registration_preserved",
    )
    changed = copy.deepcopy(manifest)
    changed["thresholds"]["selection_precision_main_effect"]["limit"] = 0.2
    assert compare_existing_registration(manifest, changed) == (
        False,
        "existing_registration_is_immutable",
    )


def test_fixture_rejects_selection_case_where_target_was_evicted():
    module, rows, _ = _valid_draft()
    drifted = copy.deepcopy(rows)
    drifted[0]["checkpoint_stream"] = [
        "target-exact",
        "noise-a",
        "noise-b",
        "noise-c",
        "noise-d",
        "noise-e",
        "noise-f",
        "noise-g",
        "noise-h",
    ]

    with pytest.raises(ValueError, match="selection_target_not_in_equal_retained_set"):
        module.validate_fixture(drifted)
