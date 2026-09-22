"""Frozen retrospective-endpoint correction for the Backblaze experiments."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "workspace/evaluation/backblaze_prediction_time_audit_v1.json"


def test_prediction_time_audit_is_frozen_and_non_scoring():
    assert sha256(AUDIT.read_bytes()).hexdigest() == (
        "750026ed4bc16e51b287e5d9ad512d6256891589d1403f346826faf8f3aec7b3")
    result = json.loads(AUDIT.read_text())
    assert all(result["checks"].values())
    assert result["cross_year_training_cases_after_cutoff"] == 2
    assert result["prospective_prediction_claim_allowed"] is False
    assert result["cross_year_complete_training_claim_allowed"] is False
    assert result["smart_values_read"] is False
    assert result["candidate_rerun"] is False
    assert result["heldout_consumed"] is False
