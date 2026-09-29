"""Equal-budget Event Memory capacity-pressure evaluation."""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval/event_memory_capacity_pressure.py"
SPEC = importlib.util.spec_from_file_location("event_memory_capacity_pressure", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_capacity_pressure_protocol_and_result_are_frozen_and_bounded():
    assert MODULE.digest(MODULE.PROTOCOL) == MODULE.PROTOCOL_SHA256
    protocol = json.loads(MODULE.PROTOCOL.read_text())
    manifest = ROOT / protocol["manifest"]
    report = MODULE.build_report(MODULE.load_rows(manifest), protocol)

    assert report["passed"] is True
    assert report["candidate_count"] == 9
    assert report["max_entries"] == 4
    assert all(value["entry_count"] == 4 for value in report["profiles"].values())
    assert all(value["aggregate_recall_fraction"] == [4, 9] for value in report["profiles"].values())
    assert len({value["total_event_cost"] for value in report["profiles"].values()}) == 1
    assert report["profiles"]["logarithmic"]["represented_age_bands"] == 3
    assert report["production_policy_changed"] is False


def test_capacity_pressure_is_deterministic():
    protocol = json.loads(MODULE.PROTOCOL.read_text())
    rows = MODULE.load_rows(ROOT / protocol["manifest"])
    assert MODULE.build_report(rows, protocol) == MODULE.build_report(rows, protocol)
