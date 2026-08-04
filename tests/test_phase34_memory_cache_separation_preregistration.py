from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from sara_engine.evaluation.phase34_separation_preregistration import (
    CASE_FAMILIES,
    EXPERIMENT_ID,
    PARENT_PROTOCOL_FINGERPRINT,
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
        / "phase34_memory_cache_separation_draft.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase34_memory_cache_separation_draft", path
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
        / "phase34_memory_cache_separation_cases.jsonl"
    )
    rows = module.load_fixture(str(fixture))
    return module, rows, module.build_draft(rows, module.environment_descriptor())


def test_followup_freezes_five_replicates_and_separation_controls():
    _, rows, draft = _valid_draft()
    manifest = build_registered_manifest(draft, managed_path=True)
    validation = validate_preregistration(manifest, managed_path=True)

    assert tuple(row["family"] for row in rows) == CASE_FAMILIES
    assert len(rows) == 12
    assert manifest["experiment_id"] == EXPERIMENT_ID
    assert manifest["parent_protocol_fingerprint"] == PARENT_PROTOCOL_FINGERPRINT
    assert tuple(manifest["replicate_seeds"]) == REPLICATE_SEEDS
    assert manifest["replicates_per_condition"] == 5
    assert manifest["budgets"]["attempted_checkpoints_per_case"] == 16
    assert validation["valid"] is True


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda draft: draft["replicate_seeds"].pop(),
            "frozen_followup_mismatch:replicate_seeds",
        ),
        (
            lambda draft: draft["budgets"].update({"max_checkpoints": 9}),
            "frozen_followup_mismatch:budgets",
        ),
        (
            lambda draft: draft["execution_policy"].update(
                {"backpropagation": True}
            ),
            "frozen_followup_mismatch:execution_policy",
        ),
        (
            lambda draft: draft.update({"parent_protocol_fingerprint": "0" * 64}),
            "frozen_followup_mismatch:parent_protocol_fingerprint",
        ),
    ],
)
def test_followup_rejects_registered_protocol_drift(mutate, error):
    _, _, draft = _valid_draft()
    mutate(draft)
    with pytest.raises(ValueError, match=error):
        build_registered_manifest(draft, managed_path=True)


def test_followup_registration_is_idempotent_and_immutable():
    _, _, draft = _valid_draft()
    manifest = build_registered_manifest(draft, managed_path=True)
    assert compare_existing_registration(manifest, manifest) == (
        True,
        "identical_registration_preserved",
    )
    changed = copy.deepcopy(manifest)
    changed["thresholds"]["pairwise_separation_rate"]["limit"] = 0.6
    assert compare_existing_registration(manifest, changed) == (
        False,
        "existing_registration_is_immutable",
    )


def test_fixture_rejects_missing_relation_and_insufficient_pressure():
    module, rows, _ = _valid_draft()
    missing = rows[:-1]
    weak = copy.deepcopy(rows)
    weak[0]["checkpoint_stream"] = weak[0]["checkpoint_stream"][:8]

    with pytest.raises(ValueError, match="case_families|expected_relation"):
        module.validate_fixture(missing)
    with pytest.raises(ValueError, match="capacity_pressure"):
        module.validate_fixture(weak)
