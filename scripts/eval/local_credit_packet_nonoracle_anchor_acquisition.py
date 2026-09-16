#!/usr/bin/env python3
"""Evaluate label-blind anchor acquisition and sparse local correction."""
from __future__ import annotations

from collections import Counter, defaultdict
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

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_nonoracle_anchor_acquisition_v1.json"
))
PROTOCOL_SHA256 = "a5253aa30dc406ec2f7505dd43cb8233afa47043cf15b630717850ea6faf5f3b"
PARENT_PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_two_change_tie_control_v1.json"
))
PARENT_RESULT = Path(workspace_path(
    "evaluation", "local_credit_packet_two_change_tie_control_development_v1.json"
))
OUTPUT = Path(workspace_path(
    "evaluation", "local_credit_packet_nonoracle_anchor_acquisition_development_v1.json"
))
ARMS = ("prior_only", "simple_tie_zero", "simple_tie_one",
        "selective_two_step", "unconditional_two_step",
        "sign_shuffle_two_step", "route_shuffle_two_step")


@dataclass(frozen=True)
class Anchor:
    source_event: int
    A_cue: int
    B_cue: int
    A_action: int
    B_action: int
    success: int
    prior_sign: int


def _hash(*parts: object) -> bytes:
    return hashlib.sha256("|".join(map(str, parts)).encode()).digest()


def _balanced_map(seed: int, cues: list[int], tag: str, zero_count: int) -> dict[int, int]:
    ordered = sorted(cues, key=lambda cue: _hash(seed, cue, tag))
    return {cue: int(position >= zero_count) for position, cue in enumerate(ordered)}


def fixture(protocol: dict, seed: int) -> dict:
    identity = protocol["identity"]
    namespace = identity["namespace"]
    A_map = _balanced_map(seed, identity["A_cues"], "A-map", 8)
    B_map = _balanced_map(seed, identity["B_cues"], "B-map", 4)
    wrong = set(sorted(identity["B_cues"], key=lambda cue: _hash(
        namespace, "initial_wrong", seed, cue
    ))[:4])
    initial = {cue: B_map[cue] ^ int(cue in wrong) for cue in B_map}
    events = []
    for A_cue in identity["A_cues"]:
        for B_cue in identity["B_cues"]:
            for A_action in (0, 1):
                for B_action in (0, 1):
                    original_index = len(events)
                    success = int(B_action == (A_map[A_cue] ^ B_map[B_cue]))
                    events.append((original_index, A_cue, B_cue, A_action,
                                   B_action, success))
    events.sort(key=lambda event: _hash(namespace, "order", seed, event[0]))
    if len(events) != 512:
        raise ValueError("Training factorial incomplete")
    first_two = defaultdict(list)
    for position, event in enumerate(events, 1):
        if len(first_two[event[1]]) < 2:
            first_two[event[1]].append((position, event[0], event[2]))
    if Counter(A_map.values()) != Counter({0: 8, 1: 8}) or any(
        len(value) != 2 for value in first_two.values()
    ):
        raise ValueError("Balanced A map or label-blind anchors unavailable")
    return {"A_map": A_map, "B_map": B_map, "initial": initial,
            "wrong_B_cues": sorted(wrong), "events": events,
            "first_two_identities": {str(cue): value for cue, value in first_two.items()}}


def _predict(A: CircuitA, cue: int, tie: int) -> int:
    zero = A.weights.get((cue, 0), 0)
    one = A.weights.get((cue, 1), 0)
    return tie if zero == one else int(one > zero)


def run_arm(data: dict, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    A = CircuitA()
    B = CircuitB()
    for cue, label in sorted(data["initial"].items()):
        B.observe_local(BLocalOutcome(cue, label))
    anchors: list[Anchor] = []
    seen = defaultdict(int)
    forward = []
    peak_state = _deep_size((A.weights, A.eligibility, B.local_votes, anchors))
    maximum_packet_bytes = 0
    for source, (original_index, A_cue, B_cue, A_action, B_action, success) in enumerate(data["events"], 1):
        forward.append((original_index, A_cue, B_cue, A_action, B_action, success))
        if seen[A_cue] < 2:
            seen[A_cue] += 1
            A.record(source, A_cue, A_action)
            receipt = A.consume(source)
            packet = B.make_packet(GlobalOutcomeToB(source, source, B_action, success),
                                   B_cue=B_cue, A_action=A_action)
            if receipt.source_event != packet.source_event:
                raise ValueError("Early source mismatch")
            A.update(A_cue, A_action, packet.sign)
            anchors.append(Anchor(source, A_cue, B_cue, A_action, B_action,
                                  success, packet.sign))
            maximum_packet_bytes = max(maximum_packet_bytes, len(packet.to_bytes()))
        peak_state = max(peak_state, _deep_size((A.weights, A.eligibility,
                                                B.local_votes, anchors)))
    if len(anchors) != 32 or set(seen.values()) != {2}:
        raise ValueError("Thirty-two source-matched anchors unavailable")
    B.local_votes.clear()
    for cue, label in sorted(data["B_map"].items()):
        B.observe_local(BLocalOutcome(cue, label))
    peak_state = max(peak_state, _deep_size((A.weights, A.eligibility,
                                            B.local_votes, anchors)))
    changed_by_cue = Counter()
    replay_lookups = replay_packets = 0
    if arm != "prior_only":
        for anchor in anchors:
            replay_lookups += 1
            packet = B.make_packet(
                GlobalOutcomeToB(anchor.source_event, anchor.source_event,
                                 anchor.B_action, anchor.success),
                B_cue=anchor.B_cue, A_action=anchor.A_action,
            )
            changed = packet.sign != anchor.prior_sign
            changed_by_cue[anchor.A_cue] += int(changed)
            if not changed and arm != "unconditional_two_step":
                continue
            magnitude = 1 if arm.startswith("simple") else 2
            packet = replace(packet, magnitude_bucket=magnitude)
            if packet.source_event != anchor.source_event or packet.causal_depth != 2:
                raise ValueError("Invalid correction packet")
            sign = -packet.sign if arm == "sign_shuffle_two_step" else packet.sign
            branch = anchor.A_action ^ int(arm == "route_shuffle_two_step")
            for _ in range(magnitude):
                A.update(anchor.A_cue, branch, sign)
            replay_packets += 1
            maximum_packet_bytes = max(maximum_packet_bytes, len(packet.to_bytes()))
            peak_state = max(peak_state, _deep_size((A.weights, A.eligibility,
                                                    B.local_votes, anchors)))
    else:
        for anchor in anchors:
            changed_by_cue[anchor.A_cue] += int(data["initial"][anchor.B_cue]
                                                != data["B_map"][anchor.B_cue])
    tie = int(arm == "simple_tie_one")
    per_cue = {}
    predictions = []
    for A_cue in sorted(data["A_map"]):
        A_action = _predict(A, A_cue, tie)
        target = data["A_map"][A_cue]
        per_cue[str(A_cue)] = {"target": target,
                               "changed_sign_count": changed_by_cue[A_cue],
                               "correct": A_action == target,
                               "tie": A.weights.get((A_cue, 0), 0)
                               == A.weights.get((A_cue, 1), 0)}
        for B_cue in sorted(data["B_map"]):
            predictions.append((A_cue, B_cue, A_action, B.forward(A_action, B_cue)))
    weights = sorted((cue, branch, weight) for (cue, branch), weight in A.weights.items())
    return {
        "accuracy": sum(row["correct"] for row in per_cue.values()) / len(per_cue),
        "per_cue": per_cue,
        "early_packets": len(anchors),
        "replay_lookups": replay_lookups,
        "replay_packets": replay_packets,
        "A_updates": A.updates,
        "B_local_updates": B.local_updates,
        "maximum_A_anchors": len(anchors),
        "maximum_B_anchors": len(anchors),
        "maximum_A_eligibility_entries": A.maximum_eligibility_entries,
        "A_features": len(A.weights),
        "B_features": len(B.local_votes),
        "maximum_packet_bytes": maximum_packet_bytes,
        "peak_state_bytes": peak_state,
        "weight_trace_sha256": hashlib.sha256(json.dumps(weights).encode()).hexdigest(),
        "forward_trace_sha256": hashlib.sha256(json.dumps(forward).encode()).hexdigest(),
        "prediction_trace_sha256": hashlib.sha256(json.dumps(predictions).encode()).hexdigest(),
    }


def _stratum(per_seed: dict, arm: str, changed: int, target: int | None = None) -> dict:
    selected = [cue for seed in per_seed.values()
                for cue in seed["arms"][arm]["per_cue"].values()
                if cue["changed_sign_count"] == changed
                and (target is None or cue["target"] == target)]
    return {"count": len(selected),
            "accuracy": sum(cue["correct"] for cue in selected) / len(selected)
            if selected else None}


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Development result already exists; refusing to rerun")
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    for path, key in ((PARENT_PROTOCOL, "parent_two_change_protocol_sha256"),
                      (PARENT_RESULT, "parent_two_change_result_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[key]:
            raise ValueError(f"Frozen parent changed: {path.name}")
    per_seed = {}
    for seed in protocol["identity"]["seeds"]:
        data = fixture(protocol, seed)
        per_seed[str(seed)] = {"first_two_identities": data["first_two_identities"],
                               "arms": {arm: run_arm(data, arm) for arm in ARMS}}
    seeds = protocol["identity"]["seeds"]
    mean = {arm: sum(per_seed[str(seed)]["arms"][arm]["accuracy"]
                     for seed in seeds) / len(seeds) for arm in ARMS}
    strata = {arm: {str(changed): {str(target): _stratum(per_seed, arm, changed, target)
                                 for target in (0, 1)} for changed in range(3)}
              for arm in ("simple_tie_zero", "simple_tie_one", "selective_two_step")}
    total_selective = sum(seed["arms"]["selective_two_step"]["replay_packets"]
                          for seed in per_seed.values())
    total_unconditional = sum(seed["arms"]["unconditional_two_step"]["replay_packets"]
                              for seed in per_seed.values())
    a = protocol["acceptance"]
    two_zero = strata["simple_tie_zero"]["2"]["0"]
    two_one = strata["simple_tie_zero"]["2"]["1"]
    checks = {
        "selective_accuracy": mean["selective_two_step"] >= a["selective_accuracy_minimum"],
        "unconditional_accuracy": mean["unconditional_two_step"] >= a["unconditional_accuracy_minimum"],
        "simple_zero_or_one_changed": all(
            strata["simple_tie_zero"][str(changed)][str(target)]["accuracy"] == a[
                "simple_zero_accuracy_by_changed_count_0_or_1"]
            for changed in (0, 1) for target in (0, 1)
            if strata["simple_tie_zero"][str(changed)][str(target)]["count"]),
        "two_change_both_targets_present": two_zero["count"] > 0 and two_one["count"] > 0,
        "simple_zero_two_change_target0": two_zero["accuracy"] == a["simple_zero_two_change_target0_accuracy"],
        "simple_zero_two_change_target1": two_one["accuracy"] == a["simple_zero_two_change_target1_accuracy"],
        "simple_one_two_change_target0": strata["simple_tie_one"]["2"]["0"]["accuracy"] == a["simple_one_two_change_target0_accuracy"],
        "simple_one_two_change_target1": strata["simple_tie_one"]["2"]["1"]["accuracy"] == a["simple_one_two_change_target1_accuracy"],
        "selective_packet_saving": total_selective < total_unconditional,
        "sign_shuffle_drop": mean["selective_two_step"] - mean["sign_shuffle_two_step"] >= a["minimum_accuracy_drop_on_wrong_sign_or_route"],
        "route_shuffle_drop": mean["selective_two_step"] - mean["route_shuffle_two_step"] >= a["minimum_accuracy_drop_on_wrong_sign_or_route"],
        "five_seed_selective": all(seed["arms"]["selective_two_step"]["accuracy"]
                                   >= a["all_five_seed_selective_accuracy_minimum"]
                                   for seed in per_seed.values()),
    }
    all_results = [result for seed in per_seed.values() for result in seed["arms"].values()]
    b = protocol["resource_budgets"]
    harness = {
        "matched_forward_trace": all(len({result["forward_trace_sha256"]
                                          for result in seed["arms"].values()}) == 1
                                     for seed in per_seed.values()),
        "first_two_label_blind": all(
            set(seed["first_two_identities"]) == {str(cue) for cue in protocol["identity"]["A_cues"]}
            and all(len(identities) == 2 for identities in seed["first_two_identities"].values())
            for seed in per_seed.values()),
        "one_step_weight_identity": all(seed["arms"]["simple_tie_zero"]["weight_trace_sha256"]
                                        == seed["arms"]["simple_tie_one"]["weight_trace_sha256"]
                                        for seed in per_seed.values()),
        "selective_unconditional_prediction_identity": all(
            seed["arms"]["selective_two_step"]["prediction_trace_sha256"]
            == seed["arms"]["unconditional_two_step"]["prediction_trace_sha256"]
            for seed in per_seed.values()),
        "strata_consistent_across_arms": all(
            len({tuple((cue, row["changed_sign_count"], row["target"])
                       for cue, row in result["per_cue"].items())
                 for result in seed["arms"].values()}) == 1
            for seed in per_seed.values()),
        "A_anchor_budget": max(result["maximum_A_anchors"] for result in all_results) <= b["maximum_A_anchors"],
        "B_anchor_budget": max(result["maximum_B_anchors"] for result in all_results) <= b["maximum_B_anchors"],
        "eligibility_budget": max(result["maximum_A_eligibility_entries"] for result in all_results) <= b["maximum_A_eligibility_entries"],
        "A_feature_budget": max(result["A_features"] for result in all_results) <= b["maximum_A_features"],
        "B_feature_budget": max(result["B_features"] for result in all_results) <= b["maximum_B_features"],
        "packet_bytes": max(result["maximum_packet_bytes"] for result in all_results) <= b["maximum_packet_bytes"],
        "replay_packet_budget": max(result["replay_packets"] for result in all_results) <= b["maximum_replay_packets"],
        "replay_lookup_budget": max(result["replay_lookups"] for result in all_results) <= b["maximum_replay_lookups"],
        "local_steps_per_packet": all(result["A_updates"] - result["early_packets"]
                                      <= result["replay_packets"] * b["maximum_local_steps_per_packet"]
                                      for result in all_results),
        "state_budget": max(result["peak_state_bytes"] for result in all_results) <= b["maximum_total_state_bytes"],
    }
    report = {
        "schema": "sara-local-credit-packet-nonoracle-anchor-acquisition-development-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "mean_accuracy": mean,
        "strata": strata,
        "selective_replay_packets": total_selective,
        "unconditional_replay_packets": total_unconditional,
        "checks": checks,
        "harness_checks": harness,
        "development_gate_passed": all(checks.values()),
        "harness_passed": all(harness.values()),
        "per_seed": per_seed,
        "heldout_consumed": False,
        "production_authorized": False,
    }
    Path(ensure_parent_directory(OUTPUT)).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mean_accuracy": mean, "strata": strata,
                      "selective_replay_packets": total_selective,
                      "unconditional_replay_packets": total_unconditional,
                      "checks": checks, "harness_checks": harness}, sort_keys=True))
    return 0 if report["development_gate_passed"] and report["harness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
