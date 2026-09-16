import json
from pathlib import Path

from sara_engine.evaluation.local_credit_packet import (
    LocalCreditPacket,
    run_credit_arm,
)


ROOT = Path(__file__).resolve().parents[1]
ROWS = (
    ROOT
    / "data"
    / "processed"
    / "benchmark_fixtures"
    / "local_credit_packet_two_stage_rows_v1.jsonl"
)


def _seed_rows(seed=955019):
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    training = [
        row for row in rows if row["seed"] == seed and row["split"] == "training"
    ]
    development = [
        row
        for row in rows
        if row["seed"] == seed and row["split"] == "development"
    ]
    return training, development


def test_packet_wire_payload_is_bounded():
    packet = LocalCreditPacket(1, 2, -1, 3, 8, 2, 3)
    assert len(packet.to_bytes()) == 21


def test_branch_addressed_credit_beats_broadcast_on_frozen_rows():
    training, development = _seed_rows()
    packet = run_credit_arm("local_credit_packet", training, development)
    broadcast = run_credit_arm("outcome_broadcast", training, development)
    assert packet["accuracy"] == 1.0
    assert broadcast["accuracy"] == 0.5
    assert packet["maximum_eligibility_entries"] == 1
    assert packet["development_updates"] == 0
    assert packet["forward_trace_sha256"] == broadcast["forward_trace_sha256"]


def test_packet_mechanism_controls_remove_the_gain():
    training, development = _seed_rows()
    intact = run_credit_arm("local_credit_packet", training, development)
    for intervention in (
        "packet_route_shuffle",
        "packet_sign_shuffle",
        "eligibility_reset_before_outcome",
        "packet_ttl_zero",
        "causal_depth_one",
    ):
        control = run_credit_arm(
            "local_credit_packet",
            training,
            development,
            intervention=intervention,
        )
        assert control["accuracy"] <= 0.5
        assert control["accuracy"] < intact["accuracy"]
