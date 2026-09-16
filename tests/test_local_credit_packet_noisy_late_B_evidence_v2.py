import hashlib
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_noisy_late_B_evidence_v2.json"
REPORT = ROOT / "workspace/evaluation/local_credit_packet_noisy_late_B_evidence_development_v2.json"
SCRIPT = ROOT / "scripts/eval/local_credit_packet_noisy_late_B_evidence_v2.py"


def test_corrected_selector_is_frozen_on_fresh_identities():
    script = runpy.run_path(str(SCRIPT))
    supplement = json.loads(PROTOCOL.read_text())
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == script["V2_SHA256"]
    protocol = json.loads(script["V1_PROTOCOL"].read_text())
    protocol["identity"] = supplement["identity"]
    assert set(protocol["identity"]["A_cues"]).isdisjoint(protocol["identity"]["B_cues"])
    for seed in protocol["identity"]["seeds"]:
        rows = script["materialize"](protocol, seed)
        training, flips = script["schedule"](protocol, seed, rows)
        script["audit_selection"](protocol, seed, rows, training, flips)
        namespace = protocol["identity"]["namespace"]
        expected_order = [row for _, row in sorted(
            enumerate(rows["training"]),
            key=lambda pair: hashlib.sha256(
                f"{namespace}|order|{seed}|{pair[0]}".encode()
            ).hexdigest(),
        )]
        assert training == expected_order
        assert len(training) == 512
        assert len(flips["quarter_noise_immediate"]) == 128
        assert len(flips["half_noise_immediate"]) == 256
        for cue in protocol["identity"]["B_cues"]:
            indices = sorted(
                (index for index, row in enumerate(training) if row[1] == cue),
                key=lambda index: hashlib.sha256(
                    f"{namespace}|flip|{seed}|{cue}|{index}".encode()
                ).hexdigest(),
            )
            assert set(indices) & flips["quarter_noise_immediate"] == set(indices[:16])
            assert set(indices) & flips["half_noise_immediate"] == set(indices[:32])


def test_corrected_development_result_and_delivery_counts():
    report = json.loads(REPORT.read_text())
    assert report["development_gate_passed"] is True
    assert report["harness_passed"] is True
    assert report["mean_packet_accuracy_by_condition"] == {
        "clean_immediate": 1.0,
        "quarter_noise_immediate": 1.0,
        "half_noise_immediate": 0.5,
        "clean_late_480": 0.90625,
        "clean_late_504": 0.575,
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
    assert report["heldout_consumed"] is False
