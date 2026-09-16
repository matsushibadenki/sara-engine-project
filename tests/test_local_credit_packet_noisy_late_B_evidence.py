import hashlib
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_noisy_late_B_evidence_v1.json"
REPORT = ROOT / "workspace/evaluation/local_credit_packet_noisy_late_B_evidence_development_v1.json"
SCRIPT = ROOT / "scripts/eval/local_credit_packet_noisy_late_B_evidence.py"


def test_noisy_late_protocol_and_label_blind_schedule():
    script = runpy.run_path(str(SCRIPT))
    protocol = json.loads(PROTOCOL.read_text())
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == script["PROTOCOL_SHA256"]
    assert set(protocol["identity"]["A_cues"]).isdisjoint(protocol["identity"]["B_cues"])
    for seed in protocol["identity"]["seeds"]:
        rows = script["materialize"](protocol, seed)
        training, flips = script["schedule"](protocol, seed, rows)
        assert len(training) == 512
        assert len(rows["development"]) == 128
        assert sum(rows["A_map"].values()) == sum(rows["B_map"].values()) == 4
        assert sorted(training) == sorted(rows["training"])
        assert len(flips["clean_immediate"]) == 0
        assert len(flips["quarter_noise_immediate"]) == 128
        assert len(flips["half_noise_immediate"]) == 256
        for B_cue in protocol["identity"]["B_cues"]:
            indices = {index for index, row in enumerate(training) if row[1] == B_cue}
            assert len(indices) == 64
            assert len(indices & flips["quarter_noise_immediate"]) == 16
            assert len(indices & flips["half_noise_immediate"]) == 32


def test_noisy_late_development_gate_and_delivery_counts():
    report = json.loads(REPORT.read_text())
    assert report["development_gate_passed"] is True
    assert report["harness_passed"] is True
    assert report["mean_packet_accuracy_by_condition"] == {
        "clean_immediate": 1.0,
        "quarter_noise_immediate": 1.0,
        "half_noise_immediate": 0.5,
        "clean_late_480": 0.875,
        "clean_late_504": 0.5,
        "clean_after_deadline_512": 0.5,
    }
    assert report["mean_quarter_noise_shortcut_accuracy"] == 0.5
    for seed in report["per_seed"].values():
        for condition, delay in (
            ("clean_immediate", 0), ("quarter_noise_immediate", 0),
            ("half_noise_immediate", 0), ("clean_late_480", 480),
            ("clean_late_504", 504), ("clean_after_deadline_512", 512),
        ):
            result = seed["conditions"][condition]["arms"]["local_credit_packet"]
            assert result["B_local_updates"] == 512 - delay
            assert result["pending_B_evidence_entries_at_scoring"] == delay
            assert result["maximum_pending_B_evidence_entries"] == delay
    assert report["heldout_consumed"] is False
