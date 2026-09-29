"""C-MAPSS availability does not bypass the reuse-rights gate."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "workspace/evaluation/cmapss_source_availability_v2.json"


def test_cmapss_availability_recheck_stays_non_acquiring_and_non_scoring():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "551c90897a947c384ec9f64290167482f6a47c84b48798e52b1d0af4a21d8714"
    )
    result = json.loads(RESULT.read_text())

    assert result["http_head"]["status"] == 200
    assert result["http_head"]["content_length"] == 12429152
    assert result["availability_gate"] == "pass"
    assert result["catalog_license"] == "License not specified"
    assert result["reuse_rights_gate"] == "blocked"
    assert result["artifact_body_downloaded"] is False
    assert result["sensor_values_read"] is False
    assert result["candidate_scored"] is False
    assert result["decision"] == "do_not_acquire_until_applicable_data_use_terms_are_explicit"
