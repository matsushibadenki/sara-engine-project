"""R2 entry audit remains metadata-only and closed when no fresh source exists."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r2_entry_audit_has_no_eligible_candidate_or_model_run():
    result = json.loads((ROOT / "workspace/evaluation/r2_entry_gate_audit_v1.json").read_text())
    assert result["decision"] == "blocked_until_new_source"
    assert result["eligible_candidates"] == []
    assert result["outcomes_read"] is False
    assert result["model_run"] is False
    assert result["candidate_scoring_authorized"] is False
    assert len(result["candidates"]) == 5


def test_consumed_and_no_delay_reasons_are_explicit():
    result = json.loads((ROOT / "workspace/evaluation/r2_entry_gate_audit_v1.json").read_text())
    statuses = {row["id"]: row["status"] for row in result["candidates"]}
    assert statuses == {
        "bpi2012": "ineligible_consumed",
        "sepsis": "ineligible_consumed",
        "beijing_air_quality": "ineligible_consumed",
        "fashion_mnist": "ineligible_no_delay",
        "ud_role_labelled": "ineligible_no_delay",
    }
