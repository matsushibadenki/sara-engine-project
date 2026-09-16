import hashlib
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_cue_complete_replay_v1.json"
REPORT = ROOT / "workspace/evaluation/local_credit_packet_cue_complete_replay_development_v1.json"
SCRIPT = ROOT / "scripts/eval/local_credit_packet_cue_complete_replay.py"


def test_cue_complete_protocol_and_anchor_opportunities():
    script = runpy.run_path(str(SCRIPT))
    protocol = json.loads(PROTOCOL.read_text())
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == script["PROTOCOL_SHA256"]
    selection_protocol = json.loads(script["PARENT_PROTOCOL"].read_text())
    selection_protocol["identity"] = protocol["identity"]
    assert set(protocol["identity"]["A_cues"]).isdisjoint(protocol["identity"]["B_cues"])
    for seed in protocol["identity"]["seeds"]:
        rows = script["materialize"](selection_protocol, seed)
        training, flips = script["schedule"](selection_protocol, seed, rows)
        script["audit_selection"](selection_protocol, seed, rows, training, flips)
        assert len(training) == 512
        assert len(rows["development"]) == 128
        assert {row[0] for row in training[:504]} == set(protocol["identity"]["A_cues"])
        assert sum(rows["A_map"].values()) == sum(rows["B_map"].values()) == 4


def test_cue_complete_replay_is_bounded_but_gate_misses_frozen_margin():
    report = json.loads(REPORT.read_text())
    assert report["harness_passed"] is True
    assert report["development_gate_passed"] is False
    assert report["checks"]["no_replay_gain"] is False
    assert sum(not value for value in report["checks"].values()) == 1
    assert report["mean_accuracy"] == {
        "no_A_credit": 0.5,
        "no_replay_packet": 0.825,
        "targeted_replay_packet": 1.0,
        "targeted_replay_direct_shortcut": 0.675,
        "oracle_control": 1.0,
    }
    assert report["control_accuracy"]["replay_disabled"] == 0.825
    for seed in report["per_seed"].values():
        targeted = seed["arms"]["targeted_replay_packet"]
        assert targeted["retained_sources"] == 8
        assert targeted["replay_lookups"] == 8
        assert targeted["replay_packets"] == 8
        assert targeted["online_packets"] == 8
        assert targeted["maximum_A_anchors"] == 8
        assert targeted["maximum_B_anchors"] == 8
        assert targeted["maximum_backward_packets_per_episode"] == 2
        assert targeted["B_local_updates"] == 8
        assert targeted["B_local_accuracy"] == 1.0
    assert report["heldout_consumed"] is False
