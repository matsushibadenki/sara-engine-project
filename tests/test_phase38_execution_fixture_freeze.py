from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sara_engine.evaluation.phase38_preregistration import CASE_FAMILIES, OPERATORS


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "eval" / "phase38_freeze_execution_fixtures.py"
    spec = importlib.util.spec_from_file_location("phase38_freeze", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def _decode(payload): return [json.loads(line) for line in payload.splitlines() if line.strip()]


def test_phase38_frozen_artifacts_cover_protocol_and_isolate_evaluator_labels():
    artifacts = _module().build_artifacts()
    sources, train, inputs, keys = (_decode(artifacts[name]) for name in ("sources","train","inputs","key"))
    assert len(sources) == 10
    assert len(train) == len(OPERATORS) == 11
    assert len(inputs) == len(keys) == len(CASE_FAMILIES) == 26
    assert [row["case_family"] for row in keys] == list(CASE_FAMILIES)
    candidate = json.dumps(inputs, sort_keys=True)
    assert "case_family" not in candidate and "expected_decision" not in candidate and "exact_target" not in candidate and "withheld_delta" not in candidate


def test_phase38_source_structure_and_transformation_families_are_partition_disjoint():
    sources = _decode(_module().build_artifacts()["sources"])
    train = [row for row in sources if row["partition"] == "train"]
    evaluation = [row for row in sources if row["partition"] == "evaluation"]
    for field in ("source_id","structure_family","transformation_family"):
        assert {row[field] for row in train}.isdisjoint({row[field] for row in evaluation})


def test_phase38_remove_operations_preserve_tombstones_in_reference_targets():
    module = _module(); artifacts = module.build_artifacts(); train = _decode(artifacts["train"])
    removed = [row for row in train if row["delta"]["operations"][0]["operator"] in {"REMOVE_NODE","REMOVE_RELATION"}]
    assert removed and all(row["target"]["tombstones"] for row in removed)
