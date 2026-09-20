"""C-MAPSS remains blocked before source or sensor access."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "workspace/evaluation/cmapss_source_availability_v1.json"


def test_cmapss_source_gate_is_frozen_and_non_scoring():
    assert sha256(RESULT.read_bytes()).hexdigest() == (
        "d8b1b026bd5dc159079bdfed97f1851397485940ffe04400160e4d243d6836bd")
    result = json.loads(RESULT.read_text())
    assert result["status"] == "blocked_official_download_and_license_review"
    assert result["download_bytes"] == 0
    assert result["candidate_selected"] is False
    assert result["candidate_scored"] is False
    assert result["smart_or_sensor_values_read"] is False
    assert result["heldout_consumed"] is False
