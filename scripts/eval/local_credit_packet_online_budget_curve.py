#!/usr/bin/env python3
"""Measure targeted local replay against a frozen post-teaching online budget."""
from __future__ import annotations

from dataclasses import replace
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
from scripts.eval.local_credit_packet_cue_complete_replay import (  # noqa: E402
    AAnchor, BAnchor,
)
from scripts.eval.local_credit_packet_noisy_late_B_evidence_v2 import (  # noqa: E402
    audit_selection, digest, materialize, schedule,
)

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_online_budget_curve_v1.json"
))
PROTOCOL_SHA256 = "69a021da296f2c97674dd3c2f5ad9554f8c51cab763246c1024524b2b7f5e71e"
PARENT_RUNNER = ROOT / "scripts/eval/local_credit_packet_cue_complete_replay.py"
PARENT_RESULT = Path(workspace_path(
    "evaluation", "local_credit_packet_cue_complete_replay_development_v1.json"
))
SELECTOR_RUNNER = ROOT / "scripts/eval/local_credit_packet_noisy_late_B_evidence_v2.py"
SELECTOR_PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_noisy_late_B_evidence_v1.json"
))
OUTPUT = Path(workspace_path(
    "evaluation", "local_credit_packet_online_budget_curve_development_v1.json"
))
ARMS = (
    "no_A_credit", "online_only_packet", "targeted_replay_packet",
    "targeted_replay_direct_shortcut", "oracle_control",
)
CONTROLS = (
    "B_local_map_reset_on_replay", "B_to_A_route_shuffle_on_replay",
    "B_to_A_sign_shuffle_on_replay", "A_anchor_erased_before_replay",
    "replay_disabled", "capacity_matched_direct_shortcut",
)


def run_arm(rows: dict, training: list[tuple], online_budget: int, arm: str,
            intervention: str = "none") -> dict:
    if arm not in ARMS and arm != "capacity_matched_direct_shortcut":
        raise ValueError("Unknown arm")
    if not 0 <= online_budget <= 32:
        raise ValueError("Invalid online budget")
    teacher_index = 512 - online_budget
    A = CircuitA(reserve=512 if arm == "capacity_matched_direct_shortcut" else 0)
    B = CircuitB()
    A_anchors: dict[int, AAnchor] = {}
    B_anchors: dict[int, BAnchor] = {}
    seen_A_cues: set[int] = set()
    retained = 0
    online_packets = 0
    replay_packets = 0
    replay_lookups = 0
    maximum_packet_bytes = 0
    maximum_A_anchors = maximum_B_anchors = 0
    maximum_backward_packets = 0
    forward = []
    peak_state = _deep_size((A.weights, A.eligibility, A.reserve,
                             B.local_votes, A_anchors, B_anchors))

    def teach_B() -> None:
        if len(A_anchors) != 8 or len(B_anchors) != 8:
            raise ValueError("Cue-complete anchors unavailable before teaching")
        for cue in sorted(rows["B_map"]):
            B.observe_local(BLocalOutcome(cue, rows["B_map"][cue]))

    for index, (A_cue, B_cue, A_action, B_action, success) in enumerate(training):
        if index == teacher_index:
            teach_B()
        source = index + 1
        forward.append((index, A_cue, B_cue, A_action, B_action, success))
        if index < teacher_index and A_cue not in seen_A_cues:
            seen_A_cues.add(A_cue)
            A_anchors[source] = AAnchor(source, index, A_cue, A_action)
            B_anchors[source] = BAnchor(source, B_cue, B_action, success)
            retained += 1
        maximum_A_anchors = max(maximum_A_anchors, len(A_anchors))
        maximum_B_anchors = max(maximum_B_anchors, len(B_anchors))
        emitted = 0
        if index >= teacher_index:
            A.record(source, A_cue, A_action)
            receipt = A.consume(source)
            outcome = GlobalOutcomeToB(source, source, B_action, success)
            if arm in ("online_only_packet", "targeted_replay_packet"):
                packet = B.make_packet(outcome, B_cue=B_cue, A_action=A_action)
                A.update(receipt.cue, receipt.active_branch, packet.sign)
                maximum_packet_bytes = max(maximum_packet_bytes, len(packet.to_bytes()))
                online_packets += 1
                emitted = 1
            elif arm in ("targeted_replay_direct_shortcut", "capacity_matched_direct_shortcut"):
                A.update(receipt.cue, receipt.active_branch, 1 if success else -1)
                emitted = 1
            elif arm == "oracle_control":
                desired = rows["A_map"][A_cue]
                A.update(receipt.cue, receipt.active_branch,
                         1 if receipt.active_branch == desired else -1)
                emitted = 1
        maximum_backward_packets = max(maximum_backward_packets, emitted)
        peak_state = max(peak_state, _deep_size((A.weights, A.eligibility, A.reserve,
                                                 B.local_votes, A_anchors, B_anchors)))
    if teacher_index == 512:
        teach_B()
    replay_sources = list(A_anchors)
    if intervention == "A_anchor_erased_before_replay":
        A_anchors.clear()
    for source in replay_sources:
        if arm not in ("targeted_replay_packet", "targeted_replay_direct_shortcut",
                       "capacity_matched_direct_shortcut", "oracle_control"):
            break
        if intervention == "replay_disabled":
            break
        replay_lookups += 1
        A_anchor = A_anchors.get(source)
        B_anchor = B_anchors.get(source)
        if A_anchor is None or B_anchor is None:
            continue
        if A_anchor.source_event != B_anchor.source_event:
            raise ValueError("Replay source mismatch")
        if arm == "targeted_replay_packet":
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
            packet = replace(packet, age=min(255, 512 - A_anchor.episode_index))
            if packet.source_event != source or packet.causal_depth != 2:
                raise ValueError("Invalid targeted packet")
            branch = A_anchor.active_branch ^ int(
                intervention == "B_to_A_route_shuffle_on_replay")
            A.update(A_anchor.A_cue, branch, packet.sign)
            maximum_packet_bytes = max(maximum_packet_bytes, len(packet.to_bytes()))
            replay_packets += 1
        elif arm in ("targeted_replay_direct_shortcut", "capacity_matched_direct_shortcut"):
            A.update(A_anchor.A_cue, A_anchor.active_branch,
                     1 if B_anchor.success else -1)
        else:
            desired = rows["A_map"][A_anchor.A_cue]
            A.update(A_anchor.A_cue, A_anchor.active_branch,
                     1 if A_anchor.active_branch == desired else -1)
        maximum_backward_packets = max(maximum_backward_packets, 1)
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
        "A_updates": A.updates,
        "B_local_updates": B.local_updates,
        "retained_sources": retained,
        "online_packets": online_packets,
        "replay_packets": replay_packets,
        "replay_lookups": replay_lookups,
        "maximum_A_anchors": maximum_A_anchors,
        "maximum_B_anchors": maximum_B_anchors,
        "maximum_A_eligibility_entries": A.maximum_eligibility_entries,
        "maximum_backward_packets_per_tick": maximum_backward_packets,
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
        (PARENT_RUNNER, "parent_replay_runner_sha256"),
        (PARENT_RESULT, "parent_replay_result_sha256"),
        (SELECTOR_RUNNER, "parent_selector_runner_sha256"),
        (SELECTOR_PROTOCOL, "parent_selector_protocol_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[key]:
            raise ValueError(f"Frozen parent changed: {path.name}")
    selection_protocol = json.loads(SELECTOR_PROTOCOL.read_text())
    selection_protocol["identity"] = protocol["identity"]
    per_seed = {}
    for seed in protocol["identity"]["seeds"]:
        rows = materialize(selection_protocol, seed)
        training, flips = schedule(selection_protocol, seed, rows)
        audit_selection(selection_protocol, seed, rows, training, flips)
        budgets = {}
        for online_budget in protocol["teaching_and_replay"]["post_teaching_online_episode_budgets"]:
            arms = {arm: run_arm(rows, training, online_budget, arm) for arm in ARMS}
            controls = ({name: run_arm(
                rows, training, online_budget,
                "capacity_matched_direct_shortcut" if name == "capacity_matched_direct_shortcut"
                else "targeted_replay_packet",
                "none" if name == "capacity_matched_direct_shortcut" else name,
            ) for name in CONTROLS} if online_budget == 0 else {})
            budgets[str(online_budget)] = {"arms": arms, "controls": controls}
        per_seed[str(seed)] = {"budgets": budgets,
                              "training_sha256": digest(training)}
    seeds = protocol["identity"]["seeds"]
    def mean(budget: int, arm: str, metric: str, *, control: bool = False) -> float:
        values = [per_seed[str(seed)]["budgets"][str(budget)][
            "controls" if control else "arms"][arm][metric] for seed in seeds]
        return sum(values) / len(values)
    curve = {
        str(budget): {arm: mean(budget, arm, "accuracy") for arm in ARMS}
        for budget in protocol["teaching_and_replay"]["post_teaching_online_episode_budgets"]
    }
    a = protocol["acceptance"]
    checks = {
        "targeted_zero": mean(0, "targeted_replay_packet", "accuracy") >= a["minimum_targeted_packet_accuracy_at_zero"],
        "targeted_two": mean(2, "targeted_replay_packet", "accuracy") >= a["minimum_targeted_packet_accuracy_at_two"],
        "zero_gain": mean(0, "targeted_replay_packet", "accuracy") - mean(0, "online_only_packet", "accuracy") >= a["minimum_gain_over_online_only_at_zero"],
        "two_gain": mean(2, "targeted_replay_packet", "accuracy") - mean(2, "online_only_packet", "accuracy") >= a["minimum_gain_over_online_only_at_two"],
        "direct_shortcut": mean(0, "targeted_replay_direct_shortcut", "accuracy") <= a["maximum_direct_shortcut_accuracy_at_zero"],
        "oracle": mean(0, "oracle_control", "accuracy") >= a["minimum_oracle_accuracy_at_zero"],
        "B_map_reset": mean(0, "targeted_replay_packet", "accuracy") - mean(0, "B_local_map_reset_on_replay", "accuracy", control=True) >= a["minimum_B_map_reset_drop_at_zero"],
        "route_shuffle": mean(0, "targeted_replay_packet", "accuracy") - mean(0, "B_to_A_route_shuffle_on_replay", "accuracy", control=True) >= a["minimum_route_shuffle_drop_at_zero"],
        "sign_shuffle": mean(0, "targeted_replay_packet", "accuracy") - mean(0, "B_to_A_sign_shuffle_on_replay", "accuracy", control=True) >= a["minimum_sign_shuffle_drop_at_zero"],
        "anchor_erasure": mean(0, "targeted_replay_packet", "accuracy") - mean(0, "A_anchor_erased_before_replay", "accuracy", control=True) >= a["minimum_anchor_erasure_drop_at_zero"],
        "five_seed_zero_gains": all(
            per_seed[str(seed)]["budgets"]["0"]["arms"]["targeted_replay_packet"]["accuracy"]
            > per_seed[str(seed)]["budgets"]["0"]["arms"]["online_only_packet"]["accuracy"]
            for seed in seeds),
    }
    all_results = [result for seed in per_seed.values() for stage in seed["budgets"].values()
                   for result in list(stage["arms"].values()) + list(stage["controls"].values())]
    b = protocol["resource_budgets"]
    harness = {
        "matched_forward_trace": all(
            len({result["forward_trace_sha256"] for result in list(stage["arms"].values()) + list(stage["controls"].values())}) == 1
            for seed in per_seed.values() for stage in seed["budgets"].values()),
        "replay_disabled_matches_online_only": all(
            seed["budgets"]["0"]["controls"]["replay_disabled"]["prediction_trace_sha256"]
            == seed["budgets"]["0"]["arms"]["online_only_packet"]["prediction_trace_sha256"]
            for seed in per_seed.values()),
        "capacity_matched_prediction": all(
            seed["budgets"]["0"]["controls"]["capacity_matched_direct_shortcut"]["prediction_trace_sha256"]
            == seed["budgets"]["0"]["arms"]["targeted_replay_direct_shortcut"]["prediction_trace_sha256"]
            for seed in per_seed.values()),
        "capacity_state_allowance": all(
            seed["budgets"]["0"]["controls"]["capacity_matched_direct_shortcut"]["peak_state_bytes"]
            >= seed["budgets"]["0"]["arms"]["targeted_replay_packet"]["peak_state_bytes"]
            for seed in per_seed.values()),
        "cue_complete_B_teaching": all(result["B_local_accuracy"] == 1.0
                                       and result["B_local_updates"] == 8 for result in all_results),
        "A_anchor_budget": max(result["maximum_A_anchors"] for result in all_results) <= b["maximum_A_anchors"],
        "B_anchor_budget": max(result["maximum_B_anchors"] for result in all_results) <= b["maximum_B_anchors"],
        "replay_lookup_budget": max(result["replay_lookups"] for result in all_results) <= b["maximum_replay_lookups"],
        "replay_packet_budget": max(result["replay_packets"] for result in all_results) <= b["maximum_replay_packets"],
        "packet_bytes": max(result["maximum_packet_bytes"] for result in all_results) <= b["maximum_packet_bytes"],
        "one_backward_packet_per_tick": max(result["maximum_backward_packets_per_tick"] for result in all_results) <= 1,
        "eligibility_budget": max(result["maximum_A_eligibility_entries"] for result in all_results) <= b["maximum_A_eligibility_entries"],
        "A_feature_budget": max(result["A_features"] for result in all_results) <= b["maximum_A_features"],
        "B_feature_budget": max(result["B_features"] for result in all_results) <= b["maximum_B_features"],
        "state_budget": max(result["peak_state_bytes"] for result in all_results) <= b["maximum_total_state_bytes"],
        "exact_replay": all(
            run_arm(*_fresh_rows(selection_protocol, seed), 0,
                    "targeted_replay_packet")["prediction_trace_sha256"]
            == per_seed[str(seed)]["budgets"]["0"]["arms"]["targeted_replay_packet"]["prediction_trace_sha256"]
            for seed in seeds),
    }
    report = {"schema": "sara-local-credit-packet-online-budget-curve-development-v1",
              "protocol_sha256": PROTOCOL_SHA256,
              "accuracy_by_online_budget": curve,
              "zero_budget_control_accuracy": {
                  control: mean(0, control, "accuracy", control=True) for control in CONTROLS
              },
              "checks": checks, "harness_checks": harness,
              "development_gate_passed": all(checks.values()),
              "harness_passed": all(harness.values()),
              "per_seed": per_seed, "heldout_consumed": False,
              "production_authorized": False}
    output = Path(ensure_parent_directory(OUTPUT))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "accuracy_by_online_budget": curve,
                      "checks": checks, "harness_checks": harness}, sort_keys=True))
    return 0 if report["development_gate_passed"] and report["harness_passed"] else 1


def _fresh_rows(protocol: dict, seed: int) -> tuple[dict, list[tuple]]:
    rows = materialize(protocol, seed)
    training, flips = schedule(protocol, seed, rows)
    audit_selection(protocol, seed, rows, training, flips)
    return rows, training


if __name__ == "__main__":
    raise SystemExit(main())
