import hashlib
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_weak_B_supervision_v1.json"
REPORT = ROOT / "workspace/evaluation/local_credit_packet_weak_B_supervision_development_v1.json"
SCRIPT = ROOT / "scripts/eval/local_credit_packet_weak_B_supervision.py"


def test_weak_supervision_protocol_is_frozen_and_label_blind():
    namespace = runpy.run_path(str(SCRIPT))
    protocol = json.loads(PROTOCOL.read_text())
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == namespace["PROTOCOL_SHA256"]
    assert len(set(protocol["identity"]["A_cues"]) & set(protocol["identity"]["B_cues"])) == 0
    for seed in protocol["identity"]["seeds"]:
        rows = namespace["materialize"](protocol, seed)
        assert len(rows["calibration"]) == 128
        assert len(rows["training"]) == 512
        assert len(rows["development"]) == 128
        assert sum(rows["A_map"].values()) == sum(rows["B_map"].values()) == 4
        assert len(set(rows["ranked_B"])) == 8
        assert all(
            success == int(B_action == (rows["A_map"][A_cue] ^ rows["B_map"][B_cue]))
            for A_cue, B_cue, A_action, B_action, success in rows["training"]
        )
        assert set((A_cue, B_cue, A_action, B_action)
                   for A_cue, B_cue, A_action, B_action, _ in rows["training"]) == {
            (A_cue, B_cue, A_action, B_action)
            for A_cue in protocol["identity"]["A_cues"]
            for B_cue in protocol["identity"]["B_cues"]
            for A_action in (0, 1) for B_action in (0, 1)
        }


def test_weak_supervision_development_gate_and_controls():
    report = json.loads(REPORT.read_text())
    assert report["development_gate_passed"] is True
    assert report["harness_passed"] is True
    assert report["mean_packet_accuracy_by_visible_B_cues"] == {
        "0": 0.5, "4": 0.775, "6": 0.875, "8": 1.0,
    }
    assert report["mean_six_cue_shortcut_accuracy"] == 0.5
    assert report["heldout_consumed"] is False
