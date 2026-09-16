#!/usr/bin/env python3
"""Evaluate balanced two-change replay and tie-rule controls."""
from __future__ import annotations

from collections import Counter
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
    "benchmark_fixtures", "local_credit_packet_two_change_tie_control_v1.json"
))
PROTOCOL_SHA256 = "ae747d8ba49b2231332cd9681ff300587463c1ce293dba91ee5a4f14c7e093aa"
PARENT_DIAGNOSTIC = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_mixed_error_replay_parity_diagnostic_v1.json"
))
PARENT_RESULT = Path(workspace_path(
    "evaluation", "local_credit_packet_mixed_error_replay_parity_diagnostic_v1.json"
))
OUTPUT = Path(workspace_path(
    "evaluation", "local_credit_packet_two_change_tie_control_development_v1.json"
))
ARMS = ("prior_tie_zero", "simple_tie_zero", "simple_tie_one",
        "correction_tie_zero", "correction_tie_one",
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


def _balanced_map(seed: int, cues: list[int], tag: str) -> dict[int, int]:
    ordered = sorted(cues, key=lambda cue: _hash(seed, cue, tag))
    return {cue: int(position >= 4) for position, cue in enumerate(ordered)}


def fixture(protocol: dict, seed: int) -> dict:
    identity = protocol["identity"]
    namespace = identity["namespace"]
    A_map = _balanced_map(seed, identity["A_cues"], "A-map")
    B_map = _balanced_map(seed, identity["B_cues"], "B-map")
    wrong = sorted(identity["B_cues"], key=lambda cue: _hash(
        namespace, "initial_wrong", seed, cue
    ))[:4]
    initial = {cue: B_map[cue] ^ int(cue in wrong) for cue in B_map}
    events = []
    for A_cue in identity["A_cues"]:
        selected = sorted(wrong, key=lambda B_cue: _hash(
            namespace, "anchor", seed, A_cue, B_cue
        ))[:2]
        for index, B_cue in enumerate(selected):
            A_action = index
            B_action = _hash(namespace, "B_action", seed, A_cue, index)[-1] & 1
            target = A_map[A_cue] ^ B_map[B_cue]
            events.append((A_cue, B_cue, A_action, B_action,
                           int(B_action == target)))
    assert len(events) == 16
    assert Counter(A_map.values()) == Counter({0: 4, 1: 4})
    assert all(sum(event[0] == cue for event in events) == 2 for cue in A_map)
    return {"A_map": A_map, "B_map": B_map, "initial": initial,
            "wrong_B_cues": sorted(wrong), "events": events}


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
    anchors = []
    forward_trace = []
    peak_state = _deep_size((A.weights, A.eligibility, B.local_votes, anchors))
    maximum_packet_bytes = 0
    for source, (A_cue, B_cue, A_action, B_action, success) in enumerate(data["events"], 1):
        A.record(source, A_cue, A_action)
        receipt = A.consume(source)
        packet = B.make_packet(GlobalOutcomeToB(source, source, B_action, success),
                               B_cue=B_cue, A_action=A_action)
        if receipt.source_event != packet.source_event:
            raise ValueError("Early source mismatch")
        A.update(A_cue, A_action, packet.sign)
        anchors.append(Anchor(source, A_cue, B_cue, A_action, B_action,
                              success, packet.sign))
        forward_trace.append((A_cue, B_cue, A_action, B_action, success))
        maximum_packet_bytes = max(maximum_packet_bytes, len(packet.to_bytes()))
        peak_state = max(peak_state, _deep_size((A.weights, A.eligibility,
                                                B.local_votes, anchors)))
    B.local_votes.clear()
    for cue, label in sorted(data["B_map"].items()):
        B.observe_local(BLocalOutcome(cue, label))
    peak_state = max(peak_state, _deep_size((A.weights, A.eligibility,
                                            B.local_votes, anchors)))
    lookups = packets = changed_signs = 0
    if arm != "prior_tie_zero":
        for anchor in anchors:
            lookups += 1
            packet = B.make_packet(
                GlobalOutcomeToB(anchor.source_event, anchor.source_event,
                                 anchor.B_action, anchor.success),
                B_cue=anchor.B_cue, A_action=anchor.A_action,
            )
            if packet.sign == anchor.prior_sign:
                continue
            changed_signs += 1
            magnitude = 1 if arm.startswith("simple") else 2
            packet = replace(packet, magnitude_bucket=magnitude)
            if packet.source_event != anchor.source_event or packet.causal_depth != 2:
                raise ValueError("Invalid source or causal depth")
            sign = -packet.sign if arm == "sign_shuffle_two_step" else packet.sign
            branch = anchor.A_action ^ int(arm == "route_shuffle_two_step")
            for _ in range(magnitude):
                A.update(anchor.A_cue, branch, sign)
            packets += 1
            maximum_packet_bytes = max(maximum_packet_bytes, len(packet.to_bytes()))
            peak_state = max(peak_state, _deep_size((A.weights, A.eligibility,
                                                    B.local_votes, anchors)))
    tie = int(arm in ("simple_tie_one", "correction_tie_one"))
    per_target = Counter()
    predictions = []
    for A_cue in sorted(data["A_map"]):
        for B_cue in sorted(data["B_map"]):
            A_action = _predict(A, A_cue, tie)
            outcome = B.forward(A_action, B_cue)
            target = data["A_map"][A_cue] ^ data["B_map"][B_cue]
            per_target[(data["A_map"][A_cue], int(outcome == target))] += 1
            predictions.append((A_cue, B_cue, A_action, outcome))
    weights = sorted((cue, branch, weight) for (cue, branch), weight in A.weights.items())
    return {
        "accuracy": sum(per_target[(target, 1)] for target in (0, 1)) / 64,
        "accuracy_by_A_target": {str(target): per_target[(target, 1)] / 32
                                 for target in (0, 1)},
        "early_packets": len(anchors),
        "replay_lookups": lookups,
        "replay_packets": packets,
        "changed_signs": changed_signs,
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
        "forward_trace_sha256": hashlib.sha256(json.dumps(forward_trace).encode()).hexdigest(),
        "prediction_trace_sha256": hashlib.sha256(json.dumps(predictions).encode()).hexdigest(),
    }


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Development result already exists; refusing to rerun")
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    for path, key in ((PARENT_DIAGNOSTIC, "parent_diagnostic_protocol_sha256"),
                      (PARENT_RESULT, "parent_diagnostic_result_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[key]:
            raise ValueError(f"Frozen parent changed: {path.name}")
    per_seed = {}
    for seed in protocol["identity"]["seeds"]:
        data = fixture(protocol, seed)
        per_seed[str(seed)] = {
            "wrong_B_cues": data["wrong_B_cues"],
            "arms": {arm: run_arm(data, arm) for arm in ARMS},
        }
    seeds = protocol["identity"]["seeds"]
    summary = {arm: sum(per_seed[str(seed)]["arms"][arm]["accuracy"]
                        for seed in seeds) / len(seeds) for arm in ARMS}
    a = protocol["acceptance"]
    checks = {
        "prior": summary["prior_tie_zero"] <= a["prior_accuracy_maximum"],
        "simple_tie_zero": summary["simple_tie_zero"] == a["simple_tie_zero_accuracy"],
        "simple_tie_one": summary["simple_tie_one"] == a["simple_tie_one_accuracy"],
        "correction_tie_zero": summary["correction_tie_zero"] >= a["correction_tie_zero_accuracy_minimum"],
        "correction_tie_one": summary["correction_tie_one"] >= a["correction_tie_one_accuracy_minimum"],
        "sign_shuffle": summary["sign_shuffle_two_step"] <= a["sign_shuffle_accuracy_maximum"],
        "route_shuffle": summary["route_shuffle_two_step"] <= a["route_shuffle_accuracy_maximum"],
        "five_seed_accuracy": all(
            seed["arms"]["simple_tie_zero"]["accuracy"] == 0.5
            and seed["arms"]["simple_tie_one"]["accuracy"] == 0.5
            and seed["arms"]["correction_tie_zero"]["accuracy"] == 1.0
            and seed["arms"]["correction_tie_one"]["accuracy"] == 1.0
            for seed in per_seed.values()),
    }
    all_results = [result for seed in per_seed.values() for result in seed["arms"].values()]
    b = protocol["resource_budgets"]
    harness = {
        "matched_forward_trace": all(len({result["forward_trace_sha256"]
                                          for result in seed["arms"].values()}) == 1
                                     for seed in per_seed.values()),
        "one_step_weight_identity": all(seed["arms"]["simple_tie_zero"]["weight_trace_sha256"]
                                        == seed["arms"]["simple_tie_one"]["weight_trace_sha256"]
                                        for seed in per_seed.values()),
        "two_step_weight_identity": all(seed["arms"]["correction_tie_zero"]["weight_trace_sha256"]
                                        == seed["arms"]["correction_tie_one"]["weight_trace_sha256"]
                                        for seed in per_seed.values()),
        "replay_packet_parity": all(len({result["replay_packets"] for arm, result in seed["arms"].items()
                                         if arm != "prior_tie_zero"}) == 1
                                    for seed in per_seed.values()),
        "sixteen_changed_signs": all(result["changed_signs"] == 16
                                     for seed in per_seed.values()
                                     for arm, result in seed["arms"].items()
                                     if arm != "prior_tie_zero"),
        "balanced_per_target_tie": all(
            seed["arms"]["simple_tie_zero"]["accuracy_by_A_target"] == {"0": 1.0, "1": 0.0}
            and seed["arms"]["simple_tie_one"]["accuracy_by_A_target"] == {"0": 0.0, "1": 1.0}
            for seed in per_seed.values()),
        "A_anchor_budget": max(result["maximum_A_anchors"] for result in all_results) <= b["maximum_A_anchors"],
        "B_anchor_budget": max(result["maximum_B_anchors"] for result in all_results) <= b["maximum_B_anchors"],
        "eligibility_budget": max(result["maximum_A_eligibility_entries"] for result in all_results) <= b["maximum_A_eligibility_entries"],
        "A_feature_budget": max(result["A_features"] for result in all_results) <= b["maximum_A_features"],
        "B_feature_budget": max(result["B_features"] for result in all_results) <= b["maximum_B_features"],
        "packet_bytes": max(result["maximum_packet_bytes"] for result in all_results) <= b["maximum_packet_bytes"],
        "replay_packets": max(result["replay_packets"] for result in all_results) <= b["maximum_replay_packets"],
        "replay_lookups": max(result["replay_lookups"] for result in all_results) <= b["maximum_replay_lookups"],
        "local_steps_per_packet": max(result["A_updates"] - result["early_packets"]
                                      for result in all_results) <= 16 * b["maximum_local_steps_per_packet"],
        "state_budget": max(result["peak_state_bytes"] for result in all_results) <= b["maximum_total_state_bytes"],
    }
    report = {
        "schema": "sara-local-credit-packet-two-change-tie-control-development-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "mean_accuracy": summary,
        "checks": checks,
        "harness_checks": harness,
        "development_gate_passed": all(checks.values()),
        "harness_passed": all(harness.values()),
        "per_seed": per_seed,
        "heldout_consumed": False,
        "production_authorized": False,
    }
    Path(ensure_parent_directory(OUTPUT)).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mean_accuracy": summary, "checks": checks,
                      "harness_checks": harness}, sort_keys=True))
    return 0 if report["development_gate_passed"] and report["harness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
