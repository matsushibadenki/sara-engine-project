import hashlib
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_delayed_B_evidence_v1.json"
REPORT = ROOT / "workspace/evaluation/local_credit_packet_delayed_B_evidence_development_v1.json"
SCRIPT = ROOT / "scripts/eval/local_credit_packet_delayed_B_evidence.py"


def test_delayed_B_protocol_is_frozen_and_rows_are_balanced():
    script = runpy.run_path(str(SCRIPT))
    protocol = json.loads(PROTOCOL.read_text())
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == script["PROTOCOL_SHA256"]
    assert set(protocol["identity"]["A_cues"]).isdisjoint(protocol["identity"]["B_cues"])
    for seed in protocol["identity"]["seeds"]:
        rows = script["materialize"](protocol, seed)
        assert len(rows["training"]) == 512
        assert len(rows["development"]) == 128
        assert sum(rows["A_map"].values()) == sum(rows["B_map"].values()) == 4
        assert all(success == int(B_action == (rows["A_map"][A_cue] ^ rows["B_map"][B_cue]))
                   for A_cue, B_cue, A_action, B_action, success in rows["training"])


def test_delayed_B_evidence_delivery_and_gate():
    report = json.loads(REPORT.read_text())
    assert report["development_gate_passed"] is True
    assert report["harness_passed"] is True
    assert report["mean_packet_accuracy_by_delay"] == {
        "0": 1.0, "16": 1.0, "64": 1.0, "256": 1.0, "512": 0.5,
    }
    assert report["mean_delay_64_shortcut_accuracy"] == 0.5
    for seed in report["per_seed"].values():
        for delay in (0, 16, 64, 256, 512):
            result = seed["delays"][str(delay)]["arms"]["local_credit_packet"]
            assert result["B_local_updates"] == 512 - delay
            assert result["pending_B_evidence_entries_at_scoring"] == delay
            assert result["maximum_pending_B_evidence_entries"] == delay
    assert report["heldout_consumed"] is False
