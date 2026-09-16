import json
from pathlib import Path

from sara_engine.evaluation.local_credit_packet_three_stage import (
    EpisodicAnchor,
    run_three_stage_arm,
)


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "data" / "processed" / "benchmark_fixtures" / "local_credit_packet_three_stage_rows_v1.jsonl"
AUDIT = ROOT / "workspace" / "evaluation" / "local_credit_packet_three_stage_identifiability.json"
NAMESPACE = "local-credit-packet-three-stage-v1"


def _seed_rows(seed=966029):
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    return (
        [row for row in rows if row["seed"] == seed and row["split"] == "training"],
        [row for row in rows if row["seed"] == seed and row["split"] == "development"],
    )


def test_anchor_wire_size_is_bounded():
    anchor = EpisodicAnchor("00" * 12, "11" * 12, 7, 12, 61)
    assert len(anchor.to_bytes()) == 33


def test_frozen_materialization_audit_has_real_overlap_and_no_direct_trace():
    audit = json.loads(AUDIT.read_text())
    assert audit["passed"] is True
    assert audit["row_count"] == 2880
    assert audit["diagnostics"]["minimum_concurrent_eligibilities"] == 8
    assert audit["diagnostics"]["peak_anchor_entries"] == 8
    assert audit["diagnostics"]["direct_available_at_outcome_count"] == 0
    assert audit["diagnostics"]["final_stage_majority_accuracy"] == 0.5


def test_overlapping_anchors_recover_expired_direct_credit():
    training, development = _seed_rows()
    direct = run_three_stage_arm("direct_packet", training, development, namespace=NAMESPACE)
    replay = run_three_stage_arm("packet_targeted_replay", training, development, namespace=NAMESPACE)
    assert direct["accuracy"] == 0.5
    assert replay["accuracy"] == 1.0
    assert direct["forward_trace_sha256"] == replay["forward_trace_sha256"]
    assert replay["peak_anchor_entries"] == 8
    assert replay["expired_direct_count"] == len(training)
    assert replay["successful_replays"] == len(training)
    assert replay["maximum_anchor_lookups_per_outcome"] == 1
    assert replay["maximum_packet_bytes"] == 21
    assert replay["peak_state_bytes"] < 1_048_576


def test_wrong_anchor_and_disabled_replay_remove_gain():
    training, development = _seed_rows()
    intact = run_three_stage_arm("packet_targeted_replay", training, development, namespace=NAMESPACE)
    for intervention in (
        "replay_disabled",
        "anchor_route_shuffle",
        "anchor_context_shuffle",
        "anchor_expired",
        "causal_depth_two",
    ):
        control = run_three_stage_arm(
            "packet_targeted_replay",
            training,
            development,
            namespace=NAMESPACE,
            intervention=intervention,
        )
        assert control["accuracy"] < intact["accuracy"]
