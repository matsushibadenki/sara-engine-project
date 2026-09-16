import hashlib
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_prior_error_repair_v1.json"
REPORT = ROOT / "workspace/evaluation/local_credit_packet_prior_error_repair_development_v1.json"
SCRIPT = ROOT / "scripts/eval/local_credit_packet_prior_error_repair.py"


def test_prior_error_protocol_and_anchor_opportunities():
    script = runpy.run_path(str(SCRIPT))
    protocol = json.loads(PROTOCOL.read_text())
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == script["PROTOCOL_SHA256"]
    assert set(protocol["identity"]["A_cues"]).isdisjoint(protocol["identity"]["B_cues"])
    selector = json.loads(script["SELECTOR_PROTOCOL"].read_text())
    selector["identity"] = protocol["identity"]
    for seed in protocol["identity"]["seeds"]:
        rows = script["materialize"](selector, seed)
        training, flips = script["schedule"](selector, seed, rows)
        script["audit_selection"](selector, seed, rows, training, flips)
        assert len(training) == 512
        assert len(rows["development"]) == 128
        assert {row[0] for row in training} == set(protocol["identity"]["A_cues"])
        assert sum(rows["A_map"].values()) == sum(rows["B_map"].values()) == 4


def test_prior_error_repair_gate_and_packet_magnitude():
    report = json.loads(REPORT.read_text())
    assert report["development_gate_passed"] is True
    assert report["harness_passed"] is True
    assert report["mean_accuracy"] == {
        "wrong_credit_only": 0.0,
        "simple_replay": 0.5,
        "correction_packet": 1.0,
        "direct_outcome_correction": 0.575,
        "oracle_control": 1.0,
    }
    assert report["control_accuracy"]["magnitude_capped_at_one"] == 0.5
    assert report["control_accuracy"]["replay_disabled"] == 0.0
    assert report["control_accuracy"]["A_anchor_erased_before_replay"] == 0.0
    assert report["control_accuracy"]["B_local_map_reset_on_replay"] == 0.375
    for seed in report["per_seed"].values():
        wrong = seed["arms"]["wrong_credit_only"]
        simple = seed["arms"]["simple_replay"]
        corrected = seed["arms"]["correction_packet"]
        assert corrected["pre_revision_A_accuracy"] == 0.0
        assert corrected["early_packets"] == 8
        assert corrected["replay_packets"] == corrected["replay_lookups"] == 8
        assert corrected["prior_sign_reversals"] == 8
        assert corrected["magnitude_two_packets"] == 8
        assert simple["magnitude_two_packets"] == 0
        assert wrong["A_updates"] == 8
        assert simple["A_updates"] == 16
        assert corrected["A_updates"] == 24
        assert corrected["B_local_updates"] == 16
        assert corrected["B_local_accuracy"] == 1.0
        assert corrected["maximum_packet_bytes"] == 21
        assert corrected["maximum_A_anchors"] == corrected["maximum_B_anchors"] == 8
    assert report["heldout_consumed"] is False
