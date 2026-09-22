"""Frozen Helpdesk source and prospective feasibility gates."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data/processed/benchmark_fixtures"
EVALUATION = ROOT / "workspace/evaluation"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_official_source_and_protocols_are_pinned():
    assert digest(ROOT / "data/raw/r2_helpdesk/helpdesk_anonymized.csv") == (
        "a25beda54ee8c9e1df82b581d09165af6ba4b23cce0f257ad3163aad93fcffdc")
    assert digest(FIXTURES / "helpdesk_source_audit_protocol_v1.json") == (
        "ece79c05a10e77e04c108aa699c7ab80822a985ae9b0f38855f1dc13357bc260")
    assert digest(FIXTURES / "helpdesk_prospective_feasibility_protocol_v1.json") == (
        "d96369420df88d11453219810c91f05ebaacfd46304522c6c642fb011ff97119")
    assert digest(FIXTURES / "helpdesk_closure_timestamp_audit_protocol_v1.json") == (
        "1b0e0b90434e98569876a87268d925f308fb71bcc692afe33372c072ad8cf173")


def test_source_passes_but_prospective_task_stops_without_candidate():
    audit_path = EVALUATION / "helpdesk_source_audit_v1.json"
    result_path = EVALUATION / "helpdesk_prospective_feasibility_v1.json"
    closure_path = EVALUATION / "helpdesk_closure_timestamp_audit_v1.json"
    assert digest(audit_path) == "0152ff7b912caf5fce31da7ee142c1210e7f0663ebbc8318a7a3e416f9f78511"
    assert digest(result_path) == "1b65ccf10e730a2aedb603ee7d6b95250e7e1fb2ae5cd2a56489585c9d3dea46"
    assert digest(closure_path) == "8115ec493696894a87dddb095c520ddbcb11afbb3972ec3c95cd9ffc86824ad6"
    audit = json.loads(audit_path.read_text())
    result = json.loads(result_path.read_text())
    closure = json.loads(closure_path.read_text())
    assert audit["passed"] is True
    assert audit["case_count"] == 4167
    assert result["passed"] is False
    assert result["case_counts"]["training"]["positive"] == 195
    assert result["case_counts"]["development"]["positive"] == 95
    assert result["candidate_scored"] is False
    assert result["descriptive_features_read"] is False
    assert closure["synthetic_closure_indicator"] is False
