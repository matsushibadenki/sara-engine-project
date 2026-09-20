"""MetroPT-3 source identity and source-only audit invariants."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/r2_metropt3/metropt3.zip"
RESULT = ROOT / "workspace/evaluation/metropt3_source_audit_v2.json"


def test_metropt3_source_and_audit_are_frozen():
    assert sha256(SOURCE.read_bytes()).hexdigest() == (
        "aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a")
    assert sha256(RESULT.read_bytes()).hexdigest() == (
        "76c20434c2d0fae0b5af5b3806ae79aa71bf5db1bf158ee70724345ae9345568")
    result = json.loads(RESULT.read_text())
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["rows"] == 1_516_948
    assert result["failure_window_row_counts"] == [8663, 2360, 17315, 1622]


def test_source_audit_did_not_score_a_candidate():
    result = json.loads(RESULT.read_text())
    assert result["model_run"] is False
    assert result["candidate_scoring_authorized"] is False
    assert result["negative_gaps"] == 0
    assert result["duplicate_timestamps"] == 0
