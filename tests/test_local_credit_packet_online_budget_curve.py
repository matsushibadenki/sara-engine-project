import hashlib
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_online_budget_curve_v1.json"
REPORT = ROOT / "workspace/evaluation/local_credit_packet_online_budget_curve_development_v1.json"
SCRIPT = ROOT / "scripts/eval/local_credit_packet_online_budget_curve.py"


def test_online_budget_curve_protocol_and_fresh_materialization():
    script = runpy.run_path(str(SCRIPT))
    protocol = json.loads(PROTOCOL.read_text())
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == script["PROTOCOL_SHA256"]
    selector = json.loads(script["SELECTOR_PROTOCOL"].read_text())
    selector["identity"] = protocol["identity"]
    assert set(protocol["identity"]["A_cues"]).isdisjoint(protocol["identity"]["B_cues"])
    for seed in protocol["identity"]["seeds"]:
        rows = script["materialize"](selector, seed)
        training, flips = script["schedule"](selector, seed, rows)
        script["audit_selection"](selector, seed, rows, training, flips)
        assert len(training) == 512
        assert len(rows["development"]) == 128
        assert {row[0] for row in training[:480]} == set(protocol["identity"]["A_cues"])
        assert sum(rows["A_map"].values()) == sum(rows["B_map"].values()) == 4


def test_online_budget_curve_gate_and_bounded_replay():
    report = json.loads(REPORT.read_text())
    assert report["development_gate_passed"] is True
    assert report["harness_passed"] is True
    assert {budget: rows["online_only_packet"] for budget, rows in
            report["accuracy_by_online_budget"].items()} == {
        "0": 0.5, "2": 0.65, "4": 0.75,
        "8": 0.85, "16": 0.95, "32": 1.0,
    }
    assert all(rows["targeted_replay_packet"] == 1.0
               for rows in report["accuracy_by_online_budget"].values())
    assert report["zero_budget_control_accuracy"]["replay_disabled"] == 0.5
    assert report["zero_budget_control_accuracy"]["A_anchor_erased_before_replay"] == 0.5
    for seed in report["per_seed"].values():
        for budget, condition in seed["budgets"].items():
            replay = condition["arms"]["targeted_replay_packet"]
            online = condition["arms"]["online_only_packet"]
            assert replay["online_packets"] == online["online_packets"] == int(budget)
            assert replay["replay_packets"] == replay["replay_lookups"] == 8
            assert online["replay_packets"] == online["replay_lookups"] == 0
            assert replay["maximum_A_anchors"] == replay["maximum_B_anchors"] == 8
            assert replay["B_local_updates"] == 8
            assert replay["B_local_accuracy"] == 1.0
    assert report["heldout_consumed"] is False
