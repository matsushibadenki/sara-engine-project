from __future__ import annotations

import json
from pathlib import Path

from sara_engine.risa.structural_delta import CanonicalStructuralDeltaCodec


ROOT = Path(__file__).resolve().parents[1]


def _train():
    path = ROOT / "data" / "processed" / "benchmark_fixtures" / "phase38_structural_delta_train.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_all_eleven_frozen_operator_examples_reconstruct_and_rollback_exactly():
    codec = CanonicalStructuralDeltaCodec()
    for row in _train():
        result = codec.apply(row["base"], row["delta"])
        assert result.accepted is True, row["example_id"]
        assert result.target == row["target"], row["example_id"]
        assert codec.rollback(result) == row["base"], row["example_id"]
        assert result.durable_mutation_allowed is False


def test_remove_operators_preserve_evidence_tombstones():
    codec = CanonicalStructuralDeltaCodec()
    for row in _train():
        if row["delta"]["operations"][0]["operator"] in {"REMOVE_NODE", "REMOVE_RELATION"}:
            result = codec.apply(row["base"], row["delta"])
            assert result.target and result.target["tombstones"]
            assert result.target["tombstones"][0]["evidence_ids"]


def test_missing_stale_corrupt_cycle_and_budget_controls_fail_closed():
    path = ROOT / "data" / "processed" / "benchmark_fixtures" / "phase38_structural_delta_execution_inputs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    codec = CanonicalStructuralDeltaCodec()
    for index in (18, 21, 22, 24, 25):
        row = rows[index]
        result = codec.apply(row["base"], row["visible_delta"])
        assert result.accepted is False
        assert result.target is None and result.rollback_receipt is None
