"""Frozen source and case-isolated prospective incident manifest."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/uci_incident_prospective_manifest_protocol_v1.json"
MANIFEST = ROOT / "data/processed/benchmark_fixtures/uci_incident_prospective_manifest_v1.jsonl"
AUDIT = ROOT / "workspace/evaluation/uci_incident_source_audit_v1.json"
RESULT = ROOT / "workspace/evaluation/uci_incident_prospective_manifest_v1.json"


def test_source_and_manifest_are_frozen():
    assert sha256(AUDIT.read_bytes()).hexdigest() == (
        "4b0b71c4e9db9de3e3b591d52786aed29e5cc0f82b84d44bbbe8566ee3fee688")
    assert sha256(PROTOCOL.read_bytes()).hexdigest() == (
        "c2f95eb5e613d67a3454cf4f47568e2348ab3fe6021e0e067c284e43e07b421d")
    assert sha256(MANIFEST.read_bytes()).hexdigest() == (
        "bc2f4b0f0386a40f16179146094b06bd3448dd3323094530fedde1fe05798c54")
    assert sha256(RESULT.read_bytes()).hexdigest() == (
        "50b8d21379bacfb72272b578e3f9b010f18275357c6f5f57bf37f2f37018fcb8")


def test_prospective_split_and_feature_seal():
    result = json.loads(RESULT.read_text())
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["cases"] == 9932
    assert {key: row["selected_per_class"] for key, row in result["counts"].items()} == {
        "training": 1386, "development": 1901, "heldout": 1679}
    assert result["non_audit_attributes_read"] is False
    assert result["candidate_scored"] is False
    assert result["heldout_features_materialized"] is False


def test_manifest_cutoffs_are_case_isolated_and_before_outcome_windows():
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    assert len({row["case_id"] for row in rows}) == len(rows)
    assert all(row["opened_at"] < row["feature_cutoff"] < row["outcome_window_end"]
               for row in rows)
    assert all(set(row) == {"case_id", "split", "label", "opened_at", "feature_cutoff",
                            "outcome_window_end", "preterminal_update_count"}
               for row in rows)
