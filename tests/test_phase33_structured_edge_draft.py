from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sara_engine.evaluation.phase33_preregistration import (
    REQUIRED_CASE_FAMILIES,
    build_registered_manifest,
    validate_preregistration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "phase33_structured_edge_draft.py"
    spec = importlib.util.spec_from_file_location(
        "phase33_structured_edge_draft",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase33_fixture_and_draft_match_frozen_protocol():
    module = _load_module()
    fixture = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_fixtures"
        / "phase33_structured_edge_cases.jsonl"
    )
    rows = module.load_fixture(str(fixture))
    environment = module.environment_descriptor()
    draft = module.build_draft(rows, environment)
    registered = build_registered_manifest(draft, managed_path=True)

    assert tuple(row["family"] for row in rows) == REQUIRED_CASE_FAMILIES
    assert len(rows) == 17
    assert environment["cpu_only"] is True
    assert environment["gpu_required"] is False
    assert validate_preregistration(
        registered,
        managed_path=True,
    )["valid"] is True


def test_phase33_fixture_rejects_missing_family_and_durable_mutation():
    module = _load_module()
    fixture = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_fixtures"
        / "phase33_structured_edge_cases.jsonl"
    )
    rows = module.load_fixture(str(fixture))
    missing = rows[:-1]
    mutable = [dict(row) for row in rows]
    mutable[0] = {
        **mutable[0],
        "expected": {
            **mutable[0]["expected"],
            "durable_mutation_allowed": True,
        },
    }

    with pytest.raises(ValueError, match="case_families"):
        module.validate_fixture(missing)
    with pytest.raises(ValueError, match="durable_mutation"):
        module.validate_fixture(mutable)
