import hashlib
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_mixed_error_revision_v1.json"
REPORT = ROOT / "workspace/evaluation/local_credit_packet_mixed_error_revision_development_v1.json"
SCRIPT = ROOT / "scripts/eval/local_credit_packet_mixed_error_revision.py"


def test_mixed_error_protocol_and_label_blind_selections():
    script = runpy.run_path(str(SCRIPT))
    protocol = json.loads(PROTOCOL.read_text())
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == script["PROTOCOL_SHA256"]
    assert set(protocol["identity"]["A_cues"]).isdisjoint(protocol["identity"]["B_cues"])
    selector = json.loads(script["SELECTOR_PROTOCOL"].read_text())
    selector["identity"] = protocol["identity"]
    namespace = protocol["identity"]["namespace"]
    for seed in protocol["identity"]["seeds"]:
        rows = script["materialize"](selector, seed)
        training, flips = script["schedule"](selector, seed, rows)
        script["audit_selection"](selector, seed, rows, training, flips)
        assert len(training) == 512
        assert len(rows["development"]) == 128
        assert sum(rows["A_map"].values()) == sum(rows["B_map"].values()) == 4
        for condition in ("full", "partial_six", "one_corrupt"):
            initial, revised, selection = script["local_maps"](
                protocol, seed, rows["B_map"], condition
            )
            cues = sorted(rows["B_map"])
            expected_wrong = sorted(cues, key=lambda cue: hashlib.sha256(
                f"{namespace}|initial_wrong|{seed}|{cue}".encode()
            ).hexdigest())[:4]
            expected_visible = sorted(cues, key=lambda cue: hashlib.sha256(
                f"{namespace}|revision_visible|{seed}|{cue}".encode()
            ).hexdigest())[:6]
            expected_corrupt = min(cues, key=lambda cue: hashlib.sha256(
                f"{namespace}|revision_corrupt|{seed}|{cue}".encode()
            ).hexdigest())
            assert selection["initially_wrong_B_cues"] == sorted(expected_wrong)
            assert selection["visible_revision_B_cues"] == sorted(expected_visible)
            assert selection["corrupt_revision_B_cue"] == expected_corrupt
            assert sum(initial[cue] != rows["B_map"][cue] for cue in cues) == 4
            if condition == "full":
                assert revised == rows["B_map"]
            elif condition == "partial_six":
                assert all(revised[cue] == initial[cue] for cue in cues if cue not in expected_visible)
            else:
                assert sum(revised[cue] != rows["B_map"][cue] for cue in cues) == 1


def test_mixed_error_development_gate_and_sparse_correction():
    report = json.loads(REPORT.read_text())
    assert report["development_gate_passed"] is True
    assert report["harness_passed"] is True
    assert report["accuracy_by_condition"]["full"] == {
        "prior_credit_only": 0.575,
        "simple_replay": 0.875,
        "correction_packet": 1.0,
        "direct_outcome_correction": 0.575,
        "oracle_control": 1.0,
    }
    assert report["accuracy_by_condition"]["partial_six"]["correction_packet"] == 0.7375
    assert report["accuracy_by_condition"]["one_corrupt"]["correction_packet"] == 0.81875
    assert report["full_control_accuracy"]["magnitude_capped_at_one"] == 0.875
    assert report["full_control_accuracy"]["replay_disabled"] == 0.575
    for seed in report["per_seed"].values():
        for condition in ("full", "partial_six", "one_corrupt"):
            result = seed["conditions"][condition]["arms"]["correction_packet"]
            assert result["early_packets"] == 16
            assert result["replay_lookups"] == 16
            assert result["replay_packets"] == result["changed_signs"]
            assert result["magnitude_two_packets"] == result["changed_signs"]
            assert result["A_updates"] == 16 + 2 * result["changed_signs"]
            assert result["maximum_A_anchors"] == result["maximum_B_anchors"] == 16
            assert result["maximum_packet_bytes"] == 21
            assert result["B_local_updates"] == 16
    assert report["heldout_consumed"] is False
