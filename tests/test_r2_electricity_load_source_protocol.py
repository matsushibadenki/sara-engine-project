"""Electricity-load acquisition remains source-only and held-out sealed."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/r2_electricity_load_source_protocol_v1.json"
ACQUISITION = ROOT / "workspace/evaluation/r2_electricity_load_acquisition_v1.json"


def test_electricity_load_source_protocol_freezes_safe_boundaries():
    protocol = json.loads(PROTOCOL.read_text())
    source = protocol["source"]
    identity = protocol["proposed_identity"]
    exclusions = protocol["mandatory_exclusions"]
    boundary = protocol["access_boundary"]

    assert source["license"] == "CC BY 4.0"
    assert source["archive_path"].startswith("data/raw/")
    assert source["maximum_archive_bytes"] <= 350 * 1024 * 1024
    assert identity["training_years"] == [2011, 2012]
    assert identity["development_years"] == [2013]
    assert identity["held_out_years"] == [2014]
    assert identity["history_intervals"] == 96
    assert identity["outcome_delay_intervals"] == 4
    assert all(exclusions.values())
    assert boundary["descriptive_load_values_before_source_gate_pass"] is False
    assert boundary["held_out_load_values_before_development_adoption"] is False
    assert boundary["candidate_scoring_authorized"] is False


def test_electricity_load_acquisition_is_integrity_only():
    result = json.loads(ACQUISITION.read_text())

    assert result["protocol_sha256"] == (
        "9828a9e359c5ebfb3e6fc88f1a399b562e31a99c3d27b07c5805773101509434"
    )
    assert result["archive_bytes"] <= 350 * 1024 * 1024
    assert len(result["archive_sha256"]) == 64
    assert result["members"][0] == {
        "name": "LD2011_2014.txt",
        "uncompressed_bytes": 710998915,
    }
    assert result["zip_crc_check"] == "pass"
    assert result["primary_member_read"] is False
    assert result["descriptive_values_read"] is False
    assert result["held_out_values_read"] is False
    assert result["candidate_scored"] is False
