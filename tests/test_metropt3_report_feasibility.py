"""Frozen negative MetroPT-3 report-level feasibility result."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/metropt3_report_feasibility_v1.json"
RESULT = ROOT / "workspace/evaluation/metropt3_report_feasibility_v1.json"


def test_protocol_and_negative_result_are_frozen():
    assert sha256(PROTOCOL.read_bytes()).hexdigest() == (
        "2058feb93d760777f861348c30041b8f1da19e7689230ab5e90c5c0a8c017476")
    assert sha256(RESULT.read_bytes()).hexdigest() == (
        "2b255dc08d1c2453aff4e8a8c6bd9c67fb005a7dc2efbca013e17b3c73bacaec")
    result = json.loads(RESULT.read_text())
    assert result["feasibility_passed"] is False
    assert result["checks"]["all_windows_meet_coverage"] is False
    assert all(value for key, value in result["checks"].items()
               if key != "all_windows_meet_coverage")


def test_failure_is_only_the_first_negative_window_coverage():
    result = json.loads(RESULT.read_text())
    failed = [row for row in result["windows"] if not row["coverage_passed"]]
    assert [(row["report_id"], row["kind"], row["rows"])
            for row in failed] == [("failure_1", "negative", 739)]
    positive = [row for row in result["windows"] if row["kind"] == "positive"]
    assert all(row["coverage_passed"] for row in positive)
    assert result["sensor_values_read"] is False
    assert result["candidate_scored"] is False
    assert result["heldout_sensor_rows_materialized"] is False
