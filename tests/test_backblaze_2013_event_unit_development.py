"""Frozen Backblaze case, event, and negative development evidence."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _path(relative: str) -> Path:
    return ROOT / relative


def test_case_manifest_and_event_dataset_are_frozen_and_split_safe():
    manifest = _path("data/processed/benchmark_fixtures/backblaze_2013_case_manifest_v1.jsonl")
    dataset = _path("data/processed/benchmark_fixtures/backblaze_2013_event_unit_development_v1.jsonl")
    assert sha256(manifest.read_bytes()).hexdigest() == (
        "71bbb3625533981822865170a50a33c9e1bf4d43d23ea0878c9bb8b543f61057")
    assert sha256(dataset.read_bytes()).hexdigest() == (
        "3cd3605049965a402976eac83fe54f55766e1434f0c2a3d28621e4b5d2659dd9")
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert len(rows) == 564
    assert {row["split"] for row in rows} == {"training", "development"}
    assert all(row["history_rows"] == 30 and len(row["events"]) <= 96 for row in rows)


def test_negative_development_result_is_frozen_without_heldout_access():
    result_path = _path("workspace/evaluation/backblaze_2013_event_unit_development_v1.json")
    assert sha256(result_path.read_bytes()).hexdigest() == (
        "14b83f875bcd201a9acb0e7791c783d75760f307a9f5e8886a99b635221f20c9")
    result = json.loads(result_path.read_text())
    assert result["development_gate_passed"] is False
    assert result["heldout_consumed"] is False
    assert result["arms"]["A_scalar_local"]["balanced_accuracy"] == 0.5
    assert result["arms"]["B_compact_event"]["balanced_accuracy"] == 0.5
    assert result["arms"]["C_temporal_state"]["balanced_accuracy"] == 0.5
    assert result["checks"]["scalar_compact_prediction_equivalence"] is True


def test_posthoc_coverage_diagnostic_does_not_rerun_candidate():
    path = _path("workspace/evaluation/backblaze_2013_event_coverage_diagnostic_v1.json")
    assert sha256(path.read_bytes()).hexdigest() == (
        "f1983d2b30024b48b69f2f258ead23ced571a3d94badd4385ed5c9e8d83ff946")
    result = json.loads(path.read_text())
    assert result["candidate_rerun"] is False
    assert result["threshold_changed"] is False
    assert result["heldout_consumed"] is False
