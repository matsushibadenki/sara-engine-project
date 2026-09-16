import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_two_change_tie_control_v1.json"
SCRIPT = ROOT / "scripts/eval/local_credit_packet_two_change_tie_control.py"
REPORT = ROOT / "workspace/evaluation/local_credit_packet_two_change_tie_control_development_v1.json"


def test_two_change_protocol_and_balanced_fixture():
    module = runpy.run_path(str(SCRIPT))
    protocol = json.loads(PROTOCOL.read_text())
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == module["PROTOCOL_SHA256"]
    for seed in protocol["identity"]["seeds"]:
        data = module["fixture"](protocol, seed)
        assert Counter(data["A_map"].values()) == Counter({0: 4, 1: 4})
        assert Counter(data["B_map"].values()) == Counter({0: 4, 1: 4})
        assert sum(data["initial"][cue] != data["B_map"][cue]
                   for cue in data["B_map"]) == 4
        assert len(data["events"]) == 16
        for cue in data["A_map"]:
            events = [event for event in data["events"] if event[0] == cue]
            assert len(events) == 2
            assert {event[2] for event in events} == {0, 1}
            assert all(event[1] in data["wrong_B_cues"] for event in events)
            assert all(event[4] == int(event[3] == (data["A_map"][cue] ^ data["B_map"][event[1]]))
                       for event in events)


def test_two_change_tie_control_development_gate_and_resources():
    report = json.loads(REPORT.read_text())
    assert report["development_gate_passed"] is True
    assert report["harness_passed"] is True
    assert all(report["checks"].values())
    assert all(report["harness_checks"].values())
    assert report["mean_accuracy"] == {
        "prior_tie_zero": 0.0,
        "simple_tie_zero": 0.5,
        "simple_tie_one": 0.5,
        "correction_tie_zero": 1.0,
        "correction_tie_one": 1.0,
        "sign_shuffle_two_step": 0.0,
        "route_shuffle_two_step": 0.0,
    }
    assert all(seed["arms"]["simple_tie_zero"]["accuracy_by_A_target"] == {"0": 1.0, "1": 0.0}
               and seed["arms"]["simple_tie_one"]["accuracy_by_A_target"] == {"0": 0.0, "1": 1.0}
               for seed in report["per_seed"].values())
    assert max(row["peak_state_bytes"] for seed in report["per_seed"].values()
               for row in seed["arms"].values()) == 4128
    assert all(row["maximum_packet_bytes"] == 21
               for seed in report["per_seed"].values()
               for row in seed["arms"].values())
    assert report["heldout_consumed"] is False
    assert report["production_authorized"] is False
