from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from sara_engine.evaluation.phase34_semantic_preregistration import (
    CASE_COUNT,
    CASE_FAMILIES,
    HORIZONS,
    LANGUAGES,
    TARGET_IDS,
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
        / "phase34_semantic_delayed_recall_draft.py"
    )
    spec = importlib.util.spec_from_file_location("phase34_semantic_draft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build():
    module = _load_draft_module()
    request = module._read_json(module.DEFAULT_REQUEST)
    rows = module.build_cases(request)
    environment = module.environment_descriptor()
    draft = module.build_draft(
        rows,
        request,
        module._read_json(module.DEFAULT_LEDGER),
        module._read_json(module.DEFAULT_GATE),
        module._read_json(module.DEFAULT_PACKET),
        module._read_json(module.DEFAULT_PARENT_PREREGISTRATION),
        module._read_json(module.DEFAULT_PARENT_REPORT),
        environment,
    )
    return module, rows, draft


def test_semantic_workload_freezes_multilingual_long_horizon_cases():
    _, rows, draft = _build()
    manifest = build_registered_manifest(draft, managed_path=True)

    assert validate_preregistration(manifest, managed_path=True)["valid"] is True
    assert len(rows) == CASE_COUNT == 270
    assert Counter(
        (row["record_id"], row["language"], row["horizon"], row["family"])
        for row in rows
    ) == Counter(
        (target, language, horizon, family)
        for target in TARGET_IDS
        for language in LANGUAGES
        for horizon in HORIZONS
        for family in CASE_FAMILIES
    )
    assert all(
        row["independent_semantic_evidence"]
        == (row["family"] == "semantic_paraphrase_recall")
        for row in rows
    )
    assert manifest["claim_boundaries"]["general_language_understanding_claim_allowed"] is False
    assert manifest["execution_policy"]["answer_or_expected_decision_visible_to_candidate"] is False
    assert manifest["execution_policy"]["selector_retuning_allowed"] is False


def test_semantic_case_generation_is_deterministic_and_not_identity_scoring():
    module, rows, draft = _build()
    request = module._read_json(module.DEFAULT_REQUEST)

    assert module.build_cases(request) == rows
    assert draft["evaluation_contract"]["exact_identity_score_is_semantic_score"] is False
    assert draft["evaluation_contract"]["token_overlap_is_semantic_score"] is False
    assert all(
        row["query_text"]
        != next(
            target["stored_excerpt"]
            for target in request["targets"]
            if target["record_id"] == row["record_id"]
        )
        for row in rows
    )


def test_semantic_draft_rejects_closed_or_drifted_human_review_gate():
    module, rows, _ = _build()
    request = module._read_json(module.DEFAULT_REQUEST)
    gate = module._read_json(module.DEFAULT_GATE)
    gate["semantic_delayed_recall_preregistration_ready"] = False

    with pytest.raises(ValueError, match="human-review gate"):
        module.build_draft(
            rows,
            request,
            module._read_json(module.DEFAULT_LEDGER),
            gate,
            module._read_json(module.DEFAULT_PACKET),
            module._read_json(module.DEFAULT_PARENT_PREREGISTRATION),
            module._read_json(module.DEFAULT_PARENT_REPORT),
            module.environment_descriptor(),
        )


def test_semantic_registration_is_immutable_and_thresholds_are_distinct():
    _, _, draft = _build()
    manifest = build_registered_manifest(draft, managed_path=True)
    assert compare_existing_registration(manifest, manifest) == (
        True,
        "identical_registration_preserved",
    )
    assert "semantic_paraphrase_macro_accuracy" in manifest["thresholds"]
    assert "selection_precision_main_effect" not in manifest["thresholds"]

    changed = copy.deepcopy(manifest)
    changed["thresholds"]["semantic_paraphrase_macro_accuracy"]["limit"] = 0.5
    with pytest.raises(ValueError, match="thresholds"):
        build_registered_manifest(changed, managed_path=True)

    changed = copy.deepcopy(manifest)
    changed["claim_boundaries"]["general_semantic_memory_claim_allowed"] = True
    assert compare_existing_registration(manifest, changed) == (
        False,
        "existing_registration_is_immutable",
    )


def test_registered_semantic_workload_matches_frozen_fixture():
    fixture_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_fixtures"
        / "phase34_semantic_delayed_recall_cases.jsonl"
    )
    manifest_path = (
        PROJECT_ROOT
        / "workspace"
        / "evaluation"
        / "phase34_semantic_delayed_recall_preregistration.json"
    )
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(rows) == CASE_COUNT
    assert validate_preregistration(manifest, managed_path=True)["valid"] is True
