#!/usr/bin/env python3
"""Test bounded local correction of A updates caused by an inverted B map."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.evaluation.local_credit_packet_multihop import (  # noqa: E402
    BLocalOutcome, CircuitA, CircuitB, GlobalOutcomeToB, _deep_size,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory, processed_data_path, workspace_path,
)
from scripts.eval.local_credit_packet_noisy_late_B_evidence_v2 import (  # noqa: E402
    audit_selection, digest, materialize, schedule,
)

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_prior_error_repair_v1.json"
))
PROTOCOL_SHA256 = "12062cc5c57aa9e780fdfc90bcb369300f8c252c586ca1d359765fc00e5aee4d"
PARENT_CANDIDATE = ROOT / "scripts/eval/local_credit_packet_online_budget_curve.py"
PARENT_HELDOUT = Path(workspace_path(
    "evaluation", "local_credit_packet_online_budget_heldout_result_v1.json"
))
PARENT_PACKET = ROOT / "src/sara_engine/evaluation/local_credit_packet.py"
SELECTOR_RUNNER = ROOT / "scripts/eval/local_credit_packet_noisy_late_B_evidence_v2.py"
SELECTOR_PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_noisy_late_B_evidence_v1.json"
))
OUTPUT = Path(workspace_path(
    "evaluation", "local_credit_packet_prior_error_repair_development_v1.json"
))
ARMS = (
    "wrong_credit_only", "simple_replay", "correction_packet",
    "direct_outcome_correction", "oracle_control",
)
CONTROLS = (
    "B_local_map_reset_on_replay", "B_to_A_route_shuffle_on_replay",
    "B_to_A_sign_shuffle_on_replay", "A_anchor_erased_before_replay",
    "replay_disabled", "magnitude_capped_at_one",
    "capacity_matched_direct_correction",
)


@dataclass(frozen=True)
class AAnchor:
    source_event: int
    episode_index: int
    A_cue: int
    active_branch: int
    prior_sign: int


@dataclass(frozen=True)
class BAnchor:
    source_event: int
    B_cue: int
    B_action: int
    success: int
    prior_sign: int


def run_arm(rows: dict, training: list[tuple], arm: str,
            intervention: str = "none") -> dict:
    if arm not in ARMS and arm != "capacity_matched_direct_correction":
        raise ValueError("Unknown arm")
    if intervention != "none" and intervention not in CONTROLS:
        raise ValueError("Unknown intervention")
    A = CircuitA(reserve=512 if arm == "capacity_matched_direct_correction" else 0)
    B = CircuitB()
    for cue in sorted(rows["B_map"]):
        B.observe_local(BLocalOutcome(cue, 1 - rows["B_map"][cue]))
    A_anchors: dict[int, AAnchor] = {}
    B_anchors: dict[int, BAnchor] = {}
    seen_A_cues: set[int] = set()
    early_packets = replay_packets = replay_lookups = magnitude_two_packets = 0
    sign_reversals = 0
    maximum_packet_bytes = 0
    forward = []
    peak_state = _deep_size((A.weights, A.eligibility, A.reserve,
                             B.local_votes, A_anchors, B_anchors))
    for index, (A_cue, B_cue, A_action, B_action, success) in enumerate(training):
        source = index + 1
        forward.append((index, A_cue, B_cue, A_action, B_action, success))
        if A_cue not in seen_A_cues:
            seen_A_cues.add(A_cue)
            A.record(source, A_cue, A_action)
            receipt = A.consume(source)
            packet = B.make_packet(
                GlobalOutcomeToB(source, source, B_action, success),
                B_cue=B_cue, A_action=A_action,
            )
            if packet.source_event != receipt.source_event:
                raise ValueError("Early packet source mismatch")
            A.update(receipt.cue, receipt.active_branch, packet.sign)
            A_anchors[source] = AAnchor(source, index, A_cue, A_action, packet.sign)
            B_anchors[source] = BAnchor(source, B_cue, B_action, success, packet.sign)
            early_packets += 1
            maximum_packet_bytes = max(maximum_packet_bytes, len(packet.to_bytes()))
        peak_state = max(peak_state, _deep_size((A.weights, A.eligibility, A.reserve,
                                                 B.local_votes, A_anchors, B_anchors)))
    if len(A_anchors) != 8 or len(B_anchors) != 8:
        raise ValueError("Bounded wrong-credit anchors incomplete")
    pre_revision_A_accuracy = sum(
        int(A.predict(cue) == target) for cue, target in rows["A_map"].items()
    ) / len(rows["A_map"])
    B.local_votes.clear()
    for cue in sorted(rows["B_map"]):
        B.observe_local(BLocalOutcome(cue, rows["B_map"][cue]))
    peak_state = max(peak_state, _deep_size((A.weights, A.eligibility, A.reserve,
                                             B.local_votes, A_anchors, B_anchors)))
    replay_sources = list(A_anchors)
    if intervention == "A_anchor_erased_before_replay":
        A_anchors.clear()
    for source in replay_sources:
        if arm == "wrong_credit_only" or intervention == "replay_disabled":
            break
        replay_lookups += 1
        A_anchor = A_anchors.get(source)
        B_anchor = B_anchors.get(source)
        if A_anchor is None or B_anchor is None:
            continue
        if A_anchor.source_event != B_anchor.source_event or A_anchor.prior_sign != B_anchor.prior_sign:
            raise ValueError("Prior local receipt mismatch")
        if arm in ("simple_replay", "correction_packet"):
            packet_intervention = {
                "B_local_map_reset_on_replay": "B_local_map_reset",
                "B_to_A_sign_shuffle_on_replay": "B_to_A_sign_shuffle",
            }.get(intervention, "none")
            packet = B.make_packet(
                GlobalOutcomeToB(source, source, B_anchor.B_action, B_anchor.success),
                B_cue=B_anchor.B_cue,
                A_action=A_anchor.active_branch,
                intervention=packet_intervention,
            )
            opposite = packet.sign == -A_anchor.prior_sign
            sign_reversals += int(opposite)
            magnitude = (2 if opposite and arm == "correction_packet"
                         and intervention != "magnitude_capped_at_one" else 1)
            packet = replace(packet, magnitude_bucket=magnitude,
                             age=min(255, 512 - A_anchor.episode_index))
            if packet.source_event != source or packet.causal_depth != 2:
                raise ValueError("Invalid correction packet")
            branch = A_anchor.active_branch ^ int(
                intervention == "B_to_A_route_shuffle_on_replay")
            for _ in range(magnitude):
                A.update(A_anchor.A_cue, branch, packet.sign)
            magnitude_two_packets += int(magnitude == 2)
            maximum_packet_bytes = max(maximum_packet_bytes, len(packet.to_bytes()))
            replay_packets += 1
        elif arm in ("direct_outcome_correction", "capacity_matched_direct_correction"):
            sign = 1 if B_anchor.success else -1
            for _ in range(2):
                A.update(A_anchor.A_cue, A_anchor.active_branch, sign)
        elif arm == "oracle_control":
            desired = rows["A_map"][A_anchor.A_cue]
            sign = 1 if A_anchor.active_branch == desired else -1
            for _ in range(2):
                A.update(A_anchor.A_cue, A_anchor.active_branch, sign)
        peak_state = max(peak_state, _deep_size((A.weights, A.eligibility, A.reserve,
                                                 B.local_votes, A_anchors, B_anchors)))
    predictions = []
    global_correct = A_correct = B_correct = 0
    for A_cue, B_cue, A_target, B_target in rows["development"]:
        A_action = A.predict(A_cue)
        B_local = B.local_prediction(B_cue)
        B_action = B.forward(A_action, B_cue)
        global_correct += int(B_action == (A_target ^ B_target))
        A_correct += int(A_action == A_target)
        B_correct += int(B_local == B_target)
        predictions.append((A_cue, B_cue, A_action, B_action))
    count = len(rows["development"])
    return {
        "accuracy": global_correct / count,
        "A_accuracy": A_correct / count,
        "B_local_accuracy": B_correct / count,
        "pre_revision_A_accuracy": pre_revision_A_accuracy,
        "early_packets": early_packets,
        "replay_packets": replay_packets,
        "replay_lookups": replay_lookups,
        "magnitude_two_packets": magnitude_two_packets,
        "prior_sign_reversals": sign_reversals,
        "A_updates": A.updates,
        "B_local_updates": B.local_updates,
        "maximum_A_anchors": 8,
        "maximum_B_anchors": 8,
        "maximum_A_eligibility_entries": A.maximum_eligibility_entries,
        "maximum_packet_bytes": maximum_packet_bytes,
        "peak_state_bytes": peak_state,
        "A_features": len(A.weights),
        "B_features": len(B.local_votes),
        "forward_trace_sha256": digest(forward),
        "prediction_trace_sha256": digest(predictions),
    }


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Development result already exists; refusing to rerun")
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    for path, key in (
        (PARENT_CANDIDATE, "parent_online_budget_candidate_sha256"),
        (PARENT_HELDOUT, "parent_online_budget_heldout_result_sha256"),
        (PARENT_PACKET, "parent_packet_source_sha256"),
        (SELECTOR_RUNNER, "parent_selector_runner_sha256"),
        (SELECTOR_PROTOCOL, "parent_selector_protocol_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[key]:
            raise ValueError(f"Frozen parent changed: {path.name}")
    selector = json.loads(SELECTOR_PROTOCOL.read_text())
    selector["identity"] = protocol["identity"]
    per_seed = {}
    for seed in protocol["identity"]["seeds"]:
        rows = materialize(selector, seed)
        training, flips = schedule(selector, seed, rows)
        audit_selection(selector, seed, rows, training, flips)
        arms = {arm: run_arm(rows, training, arm) for arm in ARMS}
        controls = {name: run_arm(
            rows, training,
            "capacity_matched_direct_correction" if name == "capacity_matched_direct_correction"
            else "correction_packet",
            "none" if name == "capacity_matched_direct_correction" else name,
        ) for name in CONTROLS}
        per_seed[str(seed)] = {"arms": arms, "controls": controls,
                              "training_sha256": digest(training)}
    seeds = protocol["identity"]["seeds"]
    def mean(arm: str, metric: str, *, control: bool = False) -> float:
        values = [per_seed[str(seed)]["controls" if control else "arms"][arm][metric]
                  for seed in seeds]
        return sum(values) / len(values)
    correction = mean("correction_packet", "accuracy")
    simple = mean("simple_replay", "accuracy")
    a = protocol["acceptance"]
    checks = {
        "correction_accuracy": correction >= a["minimum_correction_accuracy"],
        "wrong_credit_negative": mean("wrong_credit_only", "accuracy") <= a["maximum_wrong_credit_only_accuracy"],
        "simple_replay_negative": simple <= a["maximum_simple_replay_accuracy"],
        "gain_over_simple": correction - simple >= a["minimum_gain_over_simple_replay"],
        "direct_outcome_negative": mean("direct_outcome_correction", "accuracy") <= a["maximum_direct_outcome_correction_accuracy"],
        "oracle": mean("oracle_control", "accuracy") >= a["minimum_oracle_accuracy"],
        "B_map_reset": correction - mean("B_local_map_reset_on_replay", "accuracy", control=True) >= a["minimum_B_map_reset_drop"],
        "route_shuffle": correction - mean("B_to_A_route_shuffle_on_replay", "accuracy", control=True) >= a["minimum_route_shuffle_drop"],
        "sign_shuffle": correction - mean("B_to_A_sign_shuffle_on_replay", "accuracy", control=True) >= a["minimum_sign_shuffle_drop"],
        "anchor_erasure": correction - mean("A_anchor_erased_before_replay", "accuracy", control=True) >= a["minimum_anchor_erasure_drop"],
        "five_seed_gains": all(
            per_seed[str(seed)]["arms"]["correction_packet"]["accuracy"]
            > per_seed[str(seed)]["arms"]["simple_replay"]["accuracy"]
            for seed in seeds),
    }
    all_results = [result for seed in per_seed.values()
                   for result in list(seed["arms"].values()) + list(seed["controls"].values())]
    b = protocol["resource_budgets"]
    harness = {
        "matched_forward_trace": all(
            len({result["forward_trace_sha256"] for result in list(seed["arms"].values()) + list(seed["controls"].values())}) == 1
            for seed in per_seed.values()),
        "wrong_updates_verified": all(result["pre_revision_A_accuracy"] == 0.0
                                      and result["early_packets"] == 8 for result in all_results),
        "cue_complete_B_revision": all(result["B_local_accuracy"] == 1.0
                                       and result["B_local_updates"] == 16 for result in all_results),
        "eight_sign_reversals": all(
            seed["arms"]["correction_packet"]["prior_sign_reversals"] == 8
            and seed["arms"]["correction_packet"]["magnitude_two_packets"] == 8
            for seed in per_seed.values()),
        "simple_magnitude_control": all(
            seed["controls"]["magnitude_capped_at_one"]["prediction_trace_sha256"]
            == seed["arms"]["simple_replay"]["prediction_trace_sha256"]
            for seed in per_seed.values()),
        "replay_disabled_matches_wrong_credit": all(
            seed["controls"]["replay_disabled"]["prediction_trace_sha256"]
            == seed["arms"]["wrong_credit_only"]["prediction_trace_sha256"]
            for seed in per_seed.values()),
        "capacity_matched_prediction": all(
            seed["controls"]["capacity_matched_direct_correction"]["prediction_trace_sha256"]
            == seed["arms"]["direct_outcome_correction"]["prediction_trace_sha256"]
            for seed in per_seed.values()),
        "capacity_state_allowance": all(
            seed["controls"]["capacity_matched_direct_correction"]["peak_state_bytes"]
            >= seed["arms"]["correction_packet"]["peak_state_bytes"]
            for seed in per_seed.values()),
        "A_anchor_budget": max(result["maximum_A_anchors"] for result in all_results) <= b["maximum_A_anchors"],
        "B_anchor_budget": max(result["maximum_B_anchors"] for result in all_results) <= b["maximum_B_anchors"],
        "one_packet_per_tick": max(result["replay_packets"] for result in all_results) <= 8,
        "replay_lookup_budget": max(result["replay_lookups"] for result in all_results) <= 8,
        "packet_bytes": max(result["maximum_packet_bytes"] for result in all_results) <= b["maximum_packet_bytes"],
        "eligibility_budget": max(result["maximum_A_eligibility_entries"] for result in all_results) <= b["maximum_A_eligibility_entries"],
        "A_feature_budget": max(result["A_features"] for result in all_results) <= b["maximum_A_features"],
        "B_feature_budget": max(result["B_features"] for result in all_results) <= b["maximum_B_features"],
        "state_budget": max(result["peak_state_bytes"] for result in all_results) <= b["maximum_total_state_bytes"],
        "exact_replay": all(
            run_arm(*_fresh_rows(selector, seed), "correction_packet")["prediction_trace_sha256"]
            == per_seed[str(seed)]["arms"]["correction_packet"]["prediction_trace_sha256"]
            for seed in seeds),
    }
    report = {"schema": "sara-local-credit-packet-prior-error-repair-development-v1",
              "protocol_sha256": PROTOCOL_SHA256,
              "mean_accuracy": {arm: mean(arm, "accuracy") for arm in ARMS},
              "control_accuracy": {control: mean(control, "accuracy", control=True)
                                   for control in CONTROLS},
              "checks": checks, "harness_checks": harness,
              "development_gate_passed": all(checks.values()),
              "harness_passed": all(harness.values()),
              "per_seed": per_seed, "heldout_consumed": False,
              "production_authorized": False}
    output = Path(ensure_parent_directory(OUTPUT))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "mean_accuracy": report["mean_accuracy"],
                      "control_accuracy": report["control_accuracy"],
                      "checks": checks, "harness_checks": harness}, sort_keys=True))
    return 0 if report["development_gate_passed"] and report["harness_passed"] else 1


def _fresh_rows(selector: dict, seed: int) -> tuple[dict, list[tuple]]:
    rows = materialize(selector, seed)
    training, flips = schedule(selector, seed, rows)
    audit_selection(selector, seed, rows, training, flips)
    return rows, training


if __name__ == "__main__":
    raise SystemExit(main())
