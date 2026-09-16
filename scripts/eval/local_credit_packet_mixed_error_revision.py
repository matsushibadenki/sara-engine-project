#!/usr/bin/env python3
"""Evaluate sparse correction after mixed prior errors and imperfect B revision."""
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
    "benchmark_fixtures", "local_credit_packet_mixed_error_revision_v1.json"
))
PROTOCOL_SHA256 = "da2505032b701762d4ecf20772bdcfe2a56ff3a83f56aa429c011af5148745a6"
PARENT_CANDIDATE = ROOT / "scripts/eval/local_credit_packet_prior_error_repair.py"
PARENT_HELDOUT = Path(workspace_path(
    "evaluation", "local_credit_packet_prior_error_repair_heldout_result_v1.json"
))
SELECTOR_RUNNER = ROOT / "scripts/eval/local_credit_packet_noisy_late_B_evidence_v2.py"
SELECTOR_PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_noisy_late_B_evidence_v1.json"
))
OUTPUT = Path(workspace_path(
    "evaluation", "local_credit_packet_mixed_error_revision_development_v1.json"
))
CONDITIONS = ("full", "partial_six", "one_corrupt")
ARMS = ("prior_credit_only", "simple_replay", "correction_packet",
        "direct_outcome_correction", "oracle_control")
CONTROLS = ("B_local_map_reset_on_replay", "B_to_A_route_shuffle_on_replay",
            "B_to_A_sign_shuffle_on_replay", "A_anchor_erased_before_replay",
            "replay_disabled", "magnitude_capped_at_one",
            "capacity_matched_direct_correction")


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


def _rank(namespace: str, label: str, seed: int, cue: int) -> str:
    return hashlib.sha256(f"{namespace}|{label}|{seed}|{cue}".encode()).hexdigest()


def local_maps(protocol: dict, seed: int, B_map: dict[int, int],
               condition: str) -> tuple[dict[int, int], dict[int, int], dict[str, object]]:
    if condition not in CONDITIONS:
        raise ValueError("Unknown revision condition")
    namespace = protocol["identity"]["namespace"]
    cues = sorted(B_map)
    initially_wrong = set(sorted(
        cues, key=lambda cue: _rank(namespace, "initial_wrong", seed, cue)
    )[:4])
    initial = {cue: B_map[cue] ^ int(cue in initially_wrong) for cue in cues}
    visible = set(sorted(
        cues, key=lambda cue: _rank(namespace, "revision_visible", seed, cue)
    )[:6])
    corrupt_cue = min(cues, key=lambda cue: _rank(namespace, "revision_corrupt", seed, cue))
    if condition == "full":
        revised = B_map.copy()
    elif condition == "partial_six":
        revised = {cue: B_map[cue] if cue in visible else initial[cue] for cue in cues}
    else:
        revised = {cue: B_map[cue] ^ int(cue == corrupt_cue) for cue in cues}
    return initial, revised, {
        "initially_wrong_B_cues": sorted(initially_wrong),
        "visible_revision_B_cues": sorted(visible),
        "corrupt_revision_B_cue": corrupt_cue,
    }


def run_arm(rows: dict, training: list[tuple], initial: dict[int, int],
            revised: dict[int, int], arm: str, intervention: str = "none") -> dict:
    if arm not in ARMS and arm != "capacity_matched_direct_correction":
        raise ValueError("Unknown arm")
    if intervention != "none" and intervention not in CONTROLS:
        raise ValueError("Unknown intervention")
    A = CircuitA(reserve=1024 if arm == "capacity_matched_direct_correction" else 0)
    B = CircuitB()
    for cue in sorted(initial):
        B.observe_local(BLocalOutcome(cue, initial[cue]))
    A_anchors: dict[int, AAnchor] = {}
    B_anchors: dict[int, BAnchor] = {}
    seen_per_A: dict[int, int] = {}
    early_packets = replay_packets = replay_lookups = magnitude_two_packets = 0
    changed_signs = 0
    maximum_packet_bytes = 0
    forward = []
    peak_state = _deep_size((A.weights, A.eligibility, A.reserve,
                             B.local_votes, A_anchors, B_anchors))
    for index, (A_cue, B_cue, A_action, B_action, success) in enumerate(training):
        source = index + 1
        forward.append((index, A_cue, B_cue, A_action, B_action, success))
        seen = seen_per_A.get(A_cue, 0)
        if seen < 2:
            seen_per_A[A_cue] = seen + 1
            A.record(source, A_cue, A_action)
            receipt = A.consume(source)
            packet = B.make_packet(
                GlobalOutcomeToB(source, source, B_action, success),
                B_cue=B_cue, A_action=A_action,
            )
            if packet.source_event != receipt.source_event:
                raise ValueError("Early source mismatch")
            A.update(receipt.cue, receipt.active_branch, packet.sign)
            A_anchors[source] = AAnchor(source, index, A_cue, A_action, packet.sign)
            B_anchors[source] = BAnchor(source, B_cue, B_action, success, packet.sign)
            early_packets += 1
            maximum_packet_bytes = max(maximum_packet_bytes, len(packet.to_bytes()))
        peak_state = max(peak_state, _deep_size((A.weights, A.eligibility, A.reserve,
                                                 B.local_votes, A_anchors, B_anchors)))
    if len(A_anchors) != 16 or len(B_anchors) != 16 or set(seen_per_A.values()) != {2}:
        raise ValueError("Sixteen source-matched anchors unavailable")
    pre_revision_A_accuracy = sum(
        int(A.predict(cue) == target) for cue, target in rows["A_map"].items()
    ) / len(rows["A_map"])
    B.local_votes.clear()
    for cue in sorted(revised):
        B.observe_local(BLocalOutcome(cue, revised[cue]))
    peak_state = max(peak_state, _deep_size((A.weights, A.eligibility, A.reserve,
                                             B.local_votes, A_anchors, B_anchors)))
    replay_sources = list(A_anchors)
    if intervention == "A_anchor_erased_before_replay":
        A_anchors.clear()
    for source in replay_sources:
        if arm == "prior_credit_only" or intervention == "replay_disabled":
            break
        replay_lookups += 1
        A_anchor = A_anchors.get(source)
        B_anchor = B_anchors.get(source)
        if A_anchor is None or B_anchor is None:
            continue
        if A_anchor.source_event != B_anchor.source_event or A_anchor.prior_sign != B_anchor.prior_sign:
            raise ValueError("Anchor receipt mismatch")
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
            if packet.sign == A_anchor.prior_sign:
                continue
            changed_signs += 1
            magnitude = 2 if arm == "correction_packet" and intervention != "magnitude_capped_at_one" else 1
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
        else:
            desired = rows["A_map"][A_anchor.A_cue]
            sign = 1 if A_anchor.active_branch == desired else -1
            if sign != A_anchor.prior_sign:
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
        "changed_signs": changed_signs,
        "magnitude_two_packets": magnitude_two_packets,
        "A_updates": A.updates,
        "B_local_updates": B.local_updates,
        "maximum_A_anchors": len(B_anchors),
        "maximum_B_anchors": len(B_anchors),
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
        (PARENT_CANDIDATE, "parent_prior_error_candidate_sha256"),
        (PARENT_HELDOUT, "parent_prior_error_heldout_result_sha256"),
        (SELECTOR_RUNNER, "parent_selector_source_sha256"),
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
        conditions = {}
        for condition in CONDITIONS:
            initial, revised, selection = local_maps(protocol, seed, rows["B_map"], condition)
            arms = {arm: run_arm(rows, training, initial, revised, arm) for arm in ARMS}
            controls = ({name: run_arm(
                rows, training, initial, revised,
                "capacity_matched_direct_correction" if name == "capacity_matched_direct_correction"
                else "correction_packet",
                "none" if name == "capacity_matched_direct_correction" else name,
            ) for name in CONTROLS} if condition == "full" else {})
            conditions[condition] = {"selection": selection, "arms": arms,
                                     "controls": controls}
        per_seed[str(seed)] = {"conditions": conditions,
                              "training_sha256": digest(training)}
    seeds = protocol["identity"]["seeds"]
    def mean(condition: str, arm: str, metric: str, *, control: bool = False) -> float:
        values = [per_seed[str(seed)]["conditions"][condition][
            "controls" if control else "arms"][arm][metric] for seed in seeds]
        return sum(values) / len(values)
    summary = {condition: {arm: mean(condition, arm, "accuracy") for arm in ARMS}
               for condition in CONDITIONS}
    control_summary = {name: mean("full", name, "accuracy", control=True)
                       for name in CONTROLS}
    correction = summary["full"]["correction_packet"]
    a = protocol["acceptance"]
    checks = {
        "full_correction": correction >= a["full_minimum_correction_accuracy"],
        "prior_gain": correction - summary["full"]["prior_credit_only"] >= a["full_minimum_gain_over_prior_credit"],
        "simple_gain": correction - summary["full"]["simple_replay"] >= a["full_minimum_gain_over_simple_replay"],
        "direct_outcome": summary["full"]["direct_outcome_correction"] <= a["full_maximum_direct_outcome_accuracy"],
        "oracle": summary["full"]["oracle_control"] >= a["full_minimum_oracle_accuracy"],
        "B_map_reset": correction - control_summary["B_local_map_reset_on_replay"] >= a["full_minimum_B_map_reset_drop"],
        "route_shuffle": correction - control_summary["B_to_A_route_shuffle_on_replay"] >= a["full_minimum_route_shuffle_drop"],
        "sign_shuffle": correction - control_summary["B_to_A_sign_shuffle_on_replay"] >= a["full_minimum_sign_shuffle_drop"],
        "anchor_erasure": correction - control_summary["A_anchor_erased_before_replay"] >= a["full_minimum_anchor_erasure_drop"],
        "full_at_least_imperfect": correction >= summary["partial_six"]["correction_packet"]
        and correction >= summary["one_corrupt"]["correction_packet"],
    }
    all_results = [result for seed in per_seed.values() for stage in seed["conditions"].values()
                   for result in list(stage["arms"].values()) + list(stage["controls"].values())]
    b = protocol["resource_budgets"]
    harness = {
        "matched_forward_trace": all(
            len({result["forward_trace_sha256"] for result in list(stage["arms"].values()) + list(stage["controls"].values())}) == 1
            for seed in per_seed.values() for stage in seed["conditions"].values()),
        "sixteen_early_updates": all(result["early_packets"] == 16 for result in all_results),
        "B_revision_updates": all(result["B_local_updates"] == 16 for result in all_results),
        "full_B_local_accuracy": all(
            result["B_local_accuracy"] == 1.0
            for seed in per_seed.values()
            for result in list(seed["conditions"]["full"]["arms"].values())
            + list(seed["conditions"]["full"]["controls"].values())),
        "magnitude_cap_matches_simple": all(
            seed["conditions"]["full"]["controls"]["magnitude_capped_at_one"]["prediction_trace_sha256"]
            == seed["conditions"]["full"]["arms"]["simple_replay"]["prediction_trace_sha256"]
            for seed in per_seed.values()),
        "replay_off_matches_prior": all(
            seed["conditions"]["full"]["controls"]["replay_disabled"]["prediction_trace_sha256"]
            == seed["conditions"]["full"]["arms"]["prior_credit_only"]["prediction_trace_sha256"]
            for seed in per_seed.values()),
        "capacity_matched_prediction": all(
            seed["conditions"]["full"]["controls"]["capacity_matched_direct_correction"]["prediction_trace_sha256"]
            == seed["conditions"]["full"]["arms"]["direct_outcome_correction"]["prediction_trace_sha256"]
            for seed in per_seed.values()),
        "capacity_state_allowance": all(
            seed["conditions"]["full"]["controls"]["capacity_matched_direct_correction"]["peak_state_bytes"]
            >= seed["conditions"]["full"]["arms"]["correction_packet"]["peak_state_bytes"]
            for seed in per_seed.values()),
        "A_anchor_budget": max(result["maximum_A_anchors"] for result in all_results) <= b["maximum_A_anchors"],
        "B_anchor_budget": max(result["maximum_B_anchors"] for result in all_results) <= b["maximum_B_anchors"],
        "replay_lookup_budget": max(result["replay_lookups"] for result in all_results) <= b["maximum_replay_lookups"],
        "replay_packet_budget": max(result["replay_packets"] for result in all_results) <= b["maximum_replay_packets"],
        "packet_bytes": max(result["maximum_packet_bytes"] for result in all_results) <= b["maximum_packet_bytes"],
        "eligibility_budget": max(result["maximum_A_eligibility_entries"] for result in all_results) <= b["maximum_A_eligibility_entries"],
        "A_feature_budget": max(result["A_features"] for result in all_results) <= b["maximum_A_features"],
        "B_feature_budget": max(result["B_features"] for result in all_results) <= b["maximum_B_features"],
        "state_budget": max(result["peak_state_bytes"] for result in all_results) <= b["maximum_total_state_bytes"],
        "exact_replay": all(
            _replay(protocol, selector, seed) == per_seed[str(seed)]["conditions"]["full"][
                "arms"]["correction_packet"]["prediction_trace_sha256"] for seed in seeds),
    }
    report = {
        "schema": "sara-local-credit-packet-mixed-error-revision-development-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "accuracy_by_condition": summary,
        "full_control_accuracy": control_summary,
        "checks": checks,
        "harness_checks": harness,
        "development_gate_passed": all(checks.values()),
        "harness_passed": all(harness.values()),
        "per_seed": per_seed,
        "heldout_consumed": False,
        "production_authorized": False,
    }
    output = Path(ensure_parent_directory(OUTPUT))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "accuracy_by_condition": summary,
                      "full_control_accuracy": control_summary,
                      "checks": checks, "harness_checks": harness}, sort_keys=True))
    return 0 if report["development_gate_passed"] and report["harness_passed"] else 1


def _replay(protocol: dict, selector: dict, seed: int) -> str:
    rows = materialize(selector, seed)
    training, flips = schedule(selector, seed, rows)
    audit_selection(selector, seed, rows, training, flips)
    initial, revised, _ = local_maps(protocol, seed, rows["B_map"], "full")
    return run_arm(rows, training, initial, revised,
                   "correction_packet")["prediction_trace_sha256"]


if __name__ == "__main__":
    raise SystemExit(main())
