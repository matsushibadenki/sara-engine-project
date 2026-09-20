"""Frozen independent 2014 and cross-year Compact Event evidence."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_2014_source_and_negative_manifest_results_are_frozen():
    source = ROOT / "workspace/evaluation/backblaze_2014_source_audit_v1.json"
    manifest = ROOT / "workspace/evaluation/backblaze_2014_case_manifest_v1.json"
    assert sha256(source.read_bytes()).hexdigest() == (
        "c05c8e8210c058927da96634f45dcebc2ebbb200e657293d4e6a95f477e97a1e")
    assert sha256(manifest.read_bytes()).hexdigest() == (
        "47c29f082b98369d0bf037e458692c50b18508f43bd0bddf721176bcf1dbe0a4")
    source_result = json.loads(source.read_text())
    manifest_result = json.loads(manifest.read_text())
    assert source_result["passed"] is True
    assert source_result["independent_failed_drives"] == 251
    assert manifest_result["passed"] is False
    assert manifest_result["counts"]["development"] == {"negative": 35, "positive": 35}


def test_cross_year_gain_is_retained_without_opening_heldout():
    path = ROOT / "workspace/evaluation/backblaze_cross_year_compact_development_v1.json"
    assert sha256(path.read_bytes()).hexdigest() == (
        "eec95da962a78f1241502eef9ba39569d8f7ac27a1b8e771248dfda49b03c413")
    result = json.loads(path.read_text())
    assert result["arms"]["candidate"]["balanced_accuracy"] > 0.58
    assert result["arms"]["candidate"]["balanced_accuracy"] > (
        result["outcome_shuffle"]["balanced_accuracy"] + 0.09)
    assert result["suppressed_routes"] == [36]
    assert result["checks"]["minimum_quality"] is False
    assert result["development_gate_passed"] is False
    assert result["heldout_consumed"] is False
    assert result["temporal_state_used"] is False
    assert result["spiking_used"] is False
