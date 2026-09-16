import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_nonoracle_anchor_acquisition_v1.json"
SCRIPT = ROOT / "scripts/eval/local_credit_packet_nonoracle_anchor_acquisition.py"
REPORT = ROOT / "workspace/evaluation/local_credit_packet_nonoracle_anchor_acquisition_development_v1.json"


def test_nonoracle_acquisition_order_is_label_blind():
    module = runpy.run_path(str(SCRIPT))
    protocol = json.loads(PROTOCOL.read_text())
    report = json.loads(REPORT.read_text())
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == module["PROTOCOL_SHA256"]
    assert report["protocol_sha256"] == module["PROTOCOL_SHA256"]
    namespace = protocol["identity"]["namespace"]
    A_cues = protocol["identity"]["A_cues"]
    B_cues = protocol["identity"]["B_cues"]
    for seed in protocol["identity"]["seeds"]:
        data = module["fixture"](protocol, seed)
        assert Counter(data["A_map"].values()) == Counter({0: 8, 1: 8})
        assert Counter(data["B_map"].values()) == Counter({0: 4, 1: 4})
        assert sum(data["initial"][cue] != data["B_map"][cue]
                   for cue in B_cues) == 4
        order = sorted(range(512), key=lambda index: hashlib.sha256(
            f"{namespace}|order|{seed}|{index}".encode()
        ).digest())
        first_two = {str(cue): [] for cue in A_cues}
        for position, index in enumerate(order, 1):
            A_cue = A_cues[index // 32]
            B_cue = B_cues[(index % 32) // 4]
            if len(first_two[str(A_cue)]) < 2:
                first_two[str(A_cue)].append([position, index, B_cue])
        assert {cue: [list(item) for item in items]
                for cue, items in data["first_two_identities"].items()} == first_two
        assert report["per_seed"][str(seed)]["first_two_identities"] == first_two


def test_nonoracle_acquisition_strata_and_packet_saving():
    report = json.loads(REPORT.read_text())
    assert report["development_gate_passed"] is True
    assert report["harness_passed"] is True
    assert all(report["checks"].values())
    assert all(report["harness_checks"].values())
    assert report["mean_accuracy"] == {
        "prior_only": 0.475,
        "simple_tie_zero": 0.9125,
        "simple_tie_one": 0.85,
        "selective_two_step": 1.0,
        "unconditional_two_step": 1.0,
        "sign_shuffle_two_step": 0.25,
        "route_shuffle_two_step": 0.25,
    }
    assert report["selective_replay_packets"] == 79
    assert report["unconditional_replay_packets"] == 160
    strata = report["strata"]
    assert [sum(strata["selective_two_step"][str(changed)][str(target)]["count"]
                for target in (0, 1)) for changed in (0, 1, 2)] == [20, 41, 19]
    assert strata["simple_tie_zero"]["2"]["0"] == {"count": 12, "accuracy": 1.0}
    assert strata["simple_tie_zero"]["2"]["1"] == {"count": 7, "accuracy": 0.0}
    assert strata["simple_tie_one"]["2"]["0"] == {"count": 12, "accuracy": 0.0}
    assert strata["simple_tie_one"]["2"]["1"] == {"count": 7, "accuracy": 1.0}
    assert all(seed["arms"]["selective_two_step"]["accuracy"] == 1.0
               for seed in report["per_seed"].values())
    assert max(row["peak_state_bytes"] for seed in report["per_seed"].values()
               for row in seed["arms"].values()) == 7128
    assert report["heldout_consumed"] is False
    assert report["production_authorized"] is False
