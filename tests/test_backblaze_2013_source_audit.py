"""Frozen Backblaze 2013 source-audit evidence and access boundaries."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_V2 = ROOT / "data/processed/benchmark_fixtures/backblaze_2013_source_audit_v2.json"
RESULT_V1 = ROOT / "workspace/evaluation/backblaze_2013_source_audit_v1.json"
RESULT_V2 = ROOT / "workspace/evaluation/backblaze_2013_source_audit_v2.json"


def test_backblaze_v1_negative_and_v2_adjudication_are_frozen():
    assert sha256(PROTOCOL_V2.read_bytes()).hexdigest() == (
        "26af53b5c247567f65c804c687b8d53f7a4b3383afd8162649c682dd4309f7f4")
    assert sha256(RESULT_V1.read_bytes()).hexdigest() == (
        "388d074c69cd404a6fe93b8aadea7dbd5ac3caa88271dff327c72943e1e845e8")
    assert sha256(RESULT_V2.read_bytes()).hexdigest() == (
        "0c02ce968bc4b768b54f9609832cbd9aa42107f0b141ae6241caaf810d73e2eb")
    v1 = json.loads(RESULT_V1.read_text())
    v2 = json.loads(RESULT_V2.read_text())
    assert v1["passed"] is False
    assert v1["failed_drive_rows_after_failure"] == 180
    assert v2["passed"] is True
    assert all(v2["checks"].values())


def test_backblaze_support_and_feature_seal_are_preserved():
    result = json.loads(RESULT_V2.read_text())
    assert result["daily_files"] == 266
    assert result["failed_drives"] == 724
    assert result["failed_drives_with_30_prior_rows"] == 597
    assert result["failures_by_split"] == {
        "training": 400, "development": 193, "heldout": 131}
    assert result["smart_values_read"] is False
    assert result["candidate_scored"] is False
    assert result["heldout_feature_rows_materialized"] is False
