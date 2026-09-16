import json
from dataclasses import fields
from pathlib import Path

from sara_engine.evaluation.local_credit_packet import LocalCreditPacket
from sara_engine.evaluation.local_credit_packet_multihop import (
    BLocalOutcome,
    CircuitA,
    CircuitB,
    GlobalOutcomeToB,
    run_multihop_arm,
)


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "data" / "processed" / "benchmark_fixtures" / "local_credit_packet_multihop_rows_v1.jsonl"
AUDIT = ROOT / "workspace" / "evaluation" / "local_credit_packet_multihop_identifiability.json"


def _seed_rows(seed=977021):
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    rows = [row for row in rows if row["seed"] == seed]
    return tuple(
        [row for row in rows if row["phase"] == phase]
        for phase in ("B_calibration", "main_training", "development")
    )


def test_private_information_audit_blocks_shortcut():
    audit = json.loads(AUDIT.read_text())
    assert audit["passed"] is True
    assert audit["row_count"] == 3840
    assert audit["diagnostics"]["direct_shortcut_A_target_majority_accuracy"] == 0.5
    assert audit["diagnostics"]["B_private_target_decoding_accuracy"] == 1.0


def test_two_local_circuits_are_trainable_and_packet_is_one_hop():
    A = CircuitA()
    B = CircuitB()
    B.observe_local(BLocalOutcome(cue=3, target=1))
    source = 42
    A.record(source, cue=2, action=0)
    packet = B.make_packet(
        GlobalOutcomeToB(source_event=source, outcome_id=77, B_action=0, success=1),
        B_cue=3,
        A_action=0,
    )
    receipt = A.consume(packet.source_event)
    A.update(receipt.cue, receipt.active_branch, packet.sign)
    assert len(packet.to_bytes()) == 21
    assert packet.causal_depth == 2
    assert [field.name for field in fields(LocalCreditPacket)] == [
        "source_event", "outcome_id", "sign", "magnitude_bucket", "age",
        "causal_depth", "confidence",
    ]
    assert A.updates == 1
    assert B.local_updates == 1


def test_packet_beats_global_and_direct_shortcuts_on_frozen_rows():
    calibration, training, development = _seed_rows()
    packet = run_multihop_arm("local_credit_packet", calibration, training, development)
    broadcast = run_multihop_arm("global_outcome_broadcast", calibration, training, development)
    shortcut = run_multihop_arm("direct_anchor_shortcut", calibration, training, development)
    assert packet["accuracy"] == 1.0
    assert broadcast["accuracy"] == shortcut["accuracy"] == 0.5
    assert packet["forward_trace_sha256"] == shortcut["forward_trace_sha256"]
    assert packet["A_feature_count"] == 16
    assert packet["B_feature_count"] == 8
    assert packet["maximum_backward_events_per_episode"] == 2
    assert packet["peak_state_bytes"] < 1_048_576


def test_B_private_map_and_packet_address_are_causally_needed():
    calibration, training, development = _seed_rows()
    intact = run_multihop_arm("local_credit_packet", calibration, training, development)
    for intervention in (
        "B_local_map_reset",
        "B_to_A_route_shuffle",
        "B_to_A_sign_shuffle",
        "packet_delivery_disabled",
    ):
        control = run_multihop_arm(
            "local_credit_packet",
            calibration,
            training,
            development,
            intervention=intervention,
        )
        assert control["accuracy"] < intact["accuracy"]
