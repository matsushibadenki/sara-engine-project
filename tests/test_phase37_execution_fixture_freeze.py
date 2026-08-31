from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sara_engine.evaluation.phase37_preregistration import REQUIRED_CASE_FAMILIES


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "eval" / "phase37_freeze_execution_cases.py"
    spec = importlib.util.spec_from_file_location("phase37_execution_freeze", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(name):
    path = ROOT / "data" / "processed" / "benchmark_fixtures" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_phase37_execution_cases_cover_registered_families_with_isolated_labels():
    module = _module()
    artifacts = module.build_execution_artifacts(
        _rows("phase37_structural_train_base.jsonl"),
        _rows("phase37_structural_evaluation_base.jsonl"),
        _rows("../autobot/phase37_structural_source_manifest.jsonl"),
    )
    assert len(artifacts["inputs"]) == len(REQUIRED_CASE_FAMILIES) == 14
    assert [row["case_family"] for row in artifacts["keys"]] == list(REQUIRED_CASE_FAMILIES)
    candidate_text = json.dumps(artifacts["inputs"], sort_keys=True)
    assert "case_family" not in candidate_text
    assert "expected_decision" not in candidate_text
    assert "withheld_edge" not in candidate_text


def test_phase37_execution_uses_only_frozen_evaluation_sources():
    evaluation = _rows("phase37_structural_evaluation_base.jsonl")
    allowed = {row["case_id"] for row in evaluation}
    module = _module()
    artifacts = module.build_execution_artifacts(
        _rows("phase37_structural_train_base.jsonl"), evaluation,
        _rows("../autobot/phase37_structural_source_manifest.jsonl"),
    )
    assert {row["source_record_id"] for row in artifacts["inputs"]} <= allowed
    assert all(row["durable_mutation_allowed"] is False for row in artifacts["inputs"])


def test_phase37_renamed_visible_and_withheld_edges_share_one_anonymous_map():
    module = _module()
    artifacts = module.build_execution_artifacts(
        _rows("phase37_structural_train_base.jsonl"),
        _rows("phase37_structural_evaluation_base.jsonl"),
        _rows("../autobot/phase37_structural_source_manifest.jsonl"),
    )
    candidate = artifacts["inputs"][0]
    key = artifacts["keys"][0]
    visible_nodes = {value for edge in candidate["visible_edges"] for value in (edge["source"], edge["target"])}
    withheld_nodes = {key["withheld_edge"]["source"], key["withheld_edge"]["target"]}
    assert visible_nodes & withheld_nodes
    assert all(node.startswith("anonymous:0:") for node in visible_nodes | withheld_nodes)
