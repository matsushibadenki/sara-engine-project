"""Frozen exploratory incident development result and boundary amendment."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data/processed/benchmark_fixtures"
EVALUATION = ROOT / "workspace/evaluation"


def test_protocol_and_features_are_pinned():
    assert sha256((FIXTURES / "uci_incident_compact_development_protocol_v1.json").read_bytes()).hexdigest() == (
        "68ad7c0435ec1ad2d5f2553994d536742218cc0a0ff180c9d52224959f2befab")
    assert sha256((FIXTURES / "uci_incident_compact_development_v1.jsonl").read_bytes()).hexdigest() == (
        "1931bb0379ceb538d2fc8fbe7fe87def93173ff20a8d4405239f8b9194c56841")
    materialization = json.loads((EVALUATION / "uci_incident_compact_materialize_v1.json").read_text())
    assert materialization["passed"] is True
    assert materialization["cases"] == 6574
    rows = [json.loads(line) for line in (FIXTURES / "uci_incident_compact_development_v1.jsonl").read_text().splitlines()]
    assert {row["split"] for row in rows} == {"training", "development"}
    assert all(len(row["tokens"]) == 9 for row in rows)


def test_negative_result_and_boundary_are_retained():
    result_path = EVALUATION / "uci_incident_compact_development_v1.json"
    assert sha256(result_path.read_bytes()).hexdigest() == (
        "724ecebe13b9f6bd4bedeace8880cee4cd66e8ecd29449842b947e7854a7d00e")
    result = json.loads(result_path.read_text())
    amendment = json.loads((EVALUATION / "uci_incident_compact_boundary_amendment_v1.json").read_text())
    assert result["passed"] is False
    assert result["candidate"]["balanced_accuracy"] < result["controls"]["cutoff_snapshot_table"]
    assert result["heldout_scored"] is False
    assert amendment["original_development_result_sha256"] == sha256(result_path.read_bytes()).hexdigest()
    assert "violated" in amendment["effect"]


def test_materializer_selects_case_before_descriptive_access():
    path = ROOT / "scripts/eval/uci_incident_compact_materialize.py"
    spec = importlib.util.spec_from_file_location("uci_incident_compact_materialize", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class GuardedRow:
        def __getitem__(self, index):
            if index != 0:
                raise AssertionError("Descriptive field accessed before case selection")
            return "outside-selected-cases"

    assert module.selected_values(GuardedRow(), {"number": 0, "incident_state": 1}, {}) is None

    identity = sha256(b"uci-incident-7day-closure-v1-2026-09-22|case|selected").hexdigest()[:24]
    selected = {identity: {"opened_at": "2016-03-01T00:00:00", "feature_cutoff": "2016-03-03T00:00:00"}}
    indices = {"number": 0, "opened_at": 1, "sys_updated_at": 2, "incident_state": 3, "impact": 4}

    class TimedGuardedRow:
        def __init__(self, updated: str, state: str, allowed: set[int]):
            self.values = ("selected", "01/03/2016 00:00", updated, state, "private")
            self.allowed = allowed
            self.read: list[int] = []

        def __getitem__(self, index):
            if index not in self.allowed:
                raise AssertionError("Future or terminal descriptive field was read")
            self.read.append(index)
            return self.values[index]

    future = TimedGuardedRow("04/03/2016 00:00", "Open", {0, 1, 2})
    assert module.selected_values(future, indices, selected) is None
    assert future.read == [0, 1, 2]
    terminal = TimedGuardedRow("03/03/2016 00:00", "Closed", {0, 1, 2, 3})
    assert module.selected_values(terminal, indices, selected) is None
    assert terminal.read == [0, 1, 2, 3]


def test_fresh_boundary_is_insufficient_without_raw_access():
    protocol = FIXTURES / "uci_incident_fresh_boundary_audit_protocol_v1.json"
    result_path = EVALUATION / "uci_incident_fresh_boundary_audit_v1.json"
    assert sha256(protocol.read_bytes()).hexdigest() == (
        "6608591d6287556a05fc447e3ce337fc040bc088592c9ce5d312392c4bce7820")
    assert sha256(result_path.read_bytes()).hexdigest() == (
        "c9aac38d3985f97b05dcf71c1b337bf06361f295346f6c2af75dc6f4d143da7f")
    result = json.loads(result_path.read_text())
    assert result["maximum_unused_month_support"] < result["minimum_total_before_balance"]
    assert result["potentially_eligible_months"] == {}
    assert result["raw_source_rows_read"] is False
    assert result["descriptive_features_read"] is False
    assert result["candidate_scored"] is False
