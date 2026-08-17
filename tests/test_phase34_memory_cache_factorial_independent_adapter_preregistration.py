from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from sara_engine.evaluation.phase34_independent_adapter_preregistration import (
    CASE_COUNT,
    CASE_FAMILIES,
    HORIZONS,
    SOURCE_DOMAINS,
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
        / "phase34_memory_cache_factorial_independent_adapter_draft.py"
    )
    spec = importlib.util.spec_from_file_location("phase34_independent_adapter_draft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build():
    module = _load_draft_module()
    rows = module._read_jsonl(module.DEFAULT_MANIFEST)
    parent = module._read_json(module.DEFAULT_PARENT_PREREGISTRATION)
    report = module._read_json(module.DEFAULT_PARENT_REPORT)
    external = module._read_json(module.DEFAULT_EXTERNAL_GATE)
    readiness = module._read_json(module.DEFAULT_READINESS_GATE)
    plan = module.build_case_plan(rows)
    environment = module.environment_descriptor()
    draft = module.build_draft(
        rows, parent, report, external, readiness, plan, environment
    )
    return module, rows, readiness, plan, draft


def test_independent_adapter_freezes_provenance_bound_case_plan_before_execution():
    _, rows, _, plan, draft = _build()
    manifest = build_registered_manifest(draft, managed_path=True)
    validation = validate_preregistration(manifest, managed_path=True)

    assert validation["valid"] is True
    assert plan["case_count"] == CASE_COUNT == 42
    assert len({case["case_id"] for case in plan["cases"]}) == CASE_COUNT
    assert Counter(
        (case["source_domain"], case["horizon"]) for case in plan["cases"]
    ) == Counter(
        {(domain, horizon): len(CASE_FAMILIES) for domain in SOURCE_DOMAINS for horizon in HORIZONS}
    )
    source_hashes = {row["material_hash"] for row in rows}
    missing = [case for case in plan["cases"] if case["family"] == "missing_identity_control"]
    assert all(case["query_material_hash"] not in source_hashes for case in missing)
    assert all(case["semantic_accuracy_claim_allowed"] is False for case in plan["cases"])
    assert manifest["claim_boundaries"]["semantic_accuracy_claim_allowed"] is False
    assert manifest["execution_policy"]["selector_retuning_allowed"] is False
    assert manifest["execution_policy"]["query_aware_retention_allowed"] is False


def test_independent_adapter_case_plan_is_deterministic():
    module, rows, _, plan, _ = _build()

    assert module.build_case_plan(rows) == plan
    assert all(len(case["stream_material_hashes"]) == 11 for case in plan["cases"] if case["horizon"] == 10)
    assert all(len(case["stream_material_hashes"]) == 16 for case in plan["cases"] if case["horizon"] in {30, 100})


def test_independent_adapter_rejects_readiness_drift():
    module, rows, readiness, plan, _ = _build()
    readiness = copy.deepcopy(readiness)
    readiness["independent_execution_ready"] = False

    with pytest.raises(ValueError, match="readiness gate"):
        module.build_draft(
            rows,
            module._read_json(module.DEFAULT_PARENT_PREREGISTRATION),
            module._read_json(module.DEFAULT_PARENT_REPORT),
            module._read_json(module.DEFAULT_EXTERNAL_GATE),
            readiness,
            plan,
            module.environment_descriptor(),
        )


def test_independent_adapter_registration_is_immutable():
    _, _, _, _, draft = _build()
    manifest = build_registered_manifest(draft, managed_path=True)
    assert compare_existing_registration(manifest, manifest) == (
        True,
        "identical_registration_preserved",
    )
    changed = copy.deepcopy(manifest)
    changed["claim_boundaries"]["semantic_accuracy_claim_allowed"] = True
    assert compare_existing_registration(manifest, changed) == (
        False,
        "existing_registration_is_immutable",
    )
    with pytest.raises(ValueError, match="claim_boundaries"):
        build_registered_manifest(changed, managed_path=True)

    changed = copy.deepcopy(manifest)
    changed["thresholds"]["selection_precision_main_effect"]["limit"] = 0.2
    with pytest.raises(ValueError, match="thresholds"):
        build_registered_manifest(changed, managed_path=True)

    changed = copy.deepcopy(manifest)
    changed["budgets"]["max_selected_checkpoints"] = 3
    with pytest.raises(ValueError, match="budgets"):
        build_registered_manifest(changed, managed_path=True)


def test_registered_v2_adapter_matches_frozen_protocol_and_plan():
    path = (
        PROJECT_ROOT
        / "workspace"
        / "evaluation"
        / "phase34_memory_cache_factorial_independent_adapter_v2_preregistration.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert validate_preregistration(manifest, managed_path=True)["valid"] is True
    assert manifest["protocol_fingerprint"] == (
        "7e4ce13ff7e0aded273a657133263ebf9c52e7d5285c3d2a341a87233bd44ec1"
    )
    assert manifest["case_plan_fingerprint"] == (
        "b0f72e3bd963ba851d341e9b2b2ac5e60846ef3052e536bf18fbfeb971a18f9f"
    )
