"""Frozen BPI 2018 source identity and negative payment feasibility gate."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data/processed/benchmark_fixtures"
EVALUATION = ROOT / "workspace/evaluation"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def test_official_source_and_protocols_are_pinned():
    assert digest(ROOT / "data/raw/r2_bpi2018/BPI_Challenge_2018.xes.gz") == (
        "7055a5310de67abb888cee26f537cde519e1ed905b0b468a314b6c319c2b30a9")
    assert digest(FIXTURES / "bpi2018_source_acquisition_v1.json") == (
        "b174d8dc2551898191b76d6a01ea5cb6c3fa84c83021e2fca0d1cd70fa1eabc0")
    assert digest(FIXTURES / "bpi2018_source_audit_protocol_v1.json") == (
        "2702eb51879b267c8172dbd24d9b1fa01dbc62b23b21dd48d9c1f66d343564c3")
    assert digest(FIXTURES / "bpi2018_payment_feasibility_protocol_v1.json") == (
        "654efe4feba4d02d9a34087b40d824079de4743d15626f309d68f871f718d707")


def test_source_passes_but_frozen_payment_task_fails():
    audit_path = EVALUATION / "bpi2018_source_audit_v1.json"
    task_path = EVALUATION / "bpi2018_payment_feasibility_v1.json"
    assert digest(audit_path) == "bf661b05d7703cdeaca6f6d3539549498dff4e787a549e76cdaf5f233ddc9c87"
    assert digest(task_path) == "32f18d55326f96f15b229ced1ecaa8cb9beec92962dbe83db74158af79db8f62"
    audit = json.loads(audit_path.read_text())
    task = json.loads(task_path.read_text())
    assert audit["passed"] is True
    assert audit["independent_cases"] == 43809
    assert audit["event_count"] == 2514266
    assert audit["other_attribute_values_read"] is False
    assert task["passed"] is False
    assert [task["case_counts"][split]["positive"] for split in ("training", "development", "heldout")] == [1, 605, 1]
    assert task["descriptive_attribute_values_read"] is False
    assert task["candidate_scored"] is False
