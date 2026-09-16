#!/usr/bin/env python3
"""Run frozen label-noise and near-deadline B-evidence development gates."""
from __future__ import annotations

from collections import deque
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
from scripts.eval.local_credit_packet_weak_B_supervision import (  # noqa: E402
    digest, materialize,
)

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_noisy_late_B_evidence_v1.json"
))
PROTOCOL_SHA256 = "601c66675dd38b01e680ca3713ac49d660ef3a362154075b39f8a521a885e05a"
PARENT = ROOT / "src/sara_engine/evaluation/local_credit_packet_multihop.py"
MATERIALIZER = ROOT / "scripts/eval/local_credit_packet_weak_B_supervision.py"
DELAYED_RUNNER = ROOT / "scripts/eval/local_credit_packet_delayed_B_evidence.py"
DELAYED_REPORT = Path(workspace_path(
    "evaluation", "local_credit_packet_delayed_B_evidence_development_v1.json"
))
OUTPUT = Path(workspace_path(
    "evaluation", "local_credit_packet_noisy_late_B_evidence_development_v1.json"
))


def _rank(namespace: str, *parts: object) -> str:
    return hashlib.sha256("|".join(map(str, (namespace, *parts))).encode()).hexdigest()


def schedule(protocol: dict, seed: int, rows: dict) -> tuple[list[tuple], dict[str, set[int]]]:
    namespace = protocol["identity"]["namespace"]
    indexed = list(enumerate(rows["training"]))
    indexed.sort(key=lambda item: _rank(namespace + "|order", seed, item[0]))
    training = [row for _, row in indexed]
    by_B_cue = {cue: [] for cue in protocol["identity"]["B_cues"]}
    for index, (_, B_cue, _, _, _) in enumerate(training):
        by_B_cue[B_cue].append(index)
    for cue, indices in by_B_cue.items():
        if len(indices) != 64:
            raise ValueError(f"Unbalanced B cue: {cue}")
        indices.sort(key=lambda index: _rank(namespace + "|flip", seed, cue, index))
    flips = {}
    for condition in protocol["schedule"]["conditions"]:
        numerator = int(condition["flip_fraction"].split("/")[0])
        flips[condition["name"]] = {
            index for indices in by_B_cue.values() for index in indices[:16 * numerator]
        }
    return training, flips


def run_arm(rows: dict, training: list[tuple], flips: set[int], delay: int,
            arm: str, intervention: str = "none") -> dict:
    A = CircuitA(reserve=512 if arm == "capacity_matched_direct_shortcut" else 0)
    B = CircuitB()
    pending: deque[tuple[int, int, int]] = deque()
    peak_state = _deep_size((A.weights, A.eligibility, A.reserve, B.local_votes, pending))
    maximum_pending = 0
    maximum_packet_bytes = 0
    forward = []
    for index, (A_cue, B_cue, A_action, B_action, success) in enumerate(training):
        while pending and pending[0][0] <= index:
            _, cue, target = pending.popleft()
            B.observe_local(BLocalOutcome(cue, target))
        source = index + 1
        A.record(source, A_cue, A_action)
        forward.append((index, A_cue, B_cue, A_action, B_action, success))
        observed_target = rows["B_map"][B_cue] ^ int(index in flips)
        if delay == 0:
            B.observe_local(BLocalOutcome(B_cue, observed_target))
        else:
            pending.append((index + delay, B_cue, observed_target))
        maximum_pending = max(maximum_pending, len(pending))
        peak_state = max(peak_state, _deep_size((A.weights, A.eligibility, A.reserve, B.local_votes, pending)))
        receipt = A.consume(source)
        outcome = GlobalOutcomeToB(source, index + 1, B_action, success)
        if arm == "local_credit_packet":
            packet = B.make_packet(outcome, B_cue=B_cue, A_action=A_action,
                                   intervention=intervention)
            if packet.source_event != receipt.source_event or packet.causal_depth != 2:
                raise ValueError("Invalid one-hop packet")
            branch = receipt.active_branch ^ int(intervention == "B_to_A_route_shuffle")
            A.update(receipt.cue, branch, packet.sign)
            maximum_packet_bytes = max(maximum_packet_bytes, len(packet.to_bytes()))
        else:
            A.update(receipt.cue, receipt.active_branch, 1 if success else -1)
        peak_state = max(peak_state, _deep_size((A.weights, A.eligibility, A.reserve, B.local_votes, pending)))
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
        "maximum_packet_bytes": maximum_packet_bytes,
        "peak_state_bytes": peak_state,
        "A_features": len(A.weights),
        "B_features": len(B.local_votes),
        "maximum_A_eligibility_entries": A.maximum_eligibility_entries,
        "maximum_pending_B_evidence_entries": maximum_pending,
        "pending_B_evidence_entries_at_scoring": len(pending),
        "forward_trace_sha256": digest(forward),
        "prediction_trace_sha256": digest(predictions),
    }


def replay_quarter_noise(protocol: dict, seed: int) -> str:
    rows = materialize(protocol, seed)
    training, flips = schedule(protocol, seed, rows)
    return run_arm(rows, training, flips["quarter_noise_immediate"], 0,
                   "local_credit_packet")["prediction_trace_sha256"]


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Development result already exists; refusing to rerun")
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Preregistered protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    for path, key in (
        (PARENT, "parent_candidate_sha256"),
        (MATERIALIZER, "parent_materializer_sha256"),
        (DELAYED_RUNNER, "parent_delayed_runner_sha256"),
        (DELAYED_REPORT, "parent_delayed_result_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[key]:
            raise ValueError(f"Frozen parent changed: {path.name}")
    per_seed = {}
    for seed in protocol["identity"]["seeds"]:
        rows = materialize(protocol, seed)
        training, flips = schedule(protocol, seed, rows)
        conditions = {}
        for condition in protocol["schedule"]["conditions"]:
            name, delay = condition["name"], condition["delay"]
            arms = {arm: run_arm(rows, training, flips[name], delay, arm)
                    for arm in protocol["arms"]}
            controls = ({control: run_arm(rows, training, flips[name], delay,
                                          "local_credit_packet", control)
                         for control in protocol["interventions_at_quarter_noise"]}
                        if name == "quarter_noise_immediate" else {})
            conditions[name] = {"flip_count": len(flips[name]),
                                "arms": arms, "controls": controls}
        per_seed[str(seed)] = {"conditions": conditions,
                              "training_sha256": digest(training),
                              "materialization_sha256": digest(rows)}
    seeds = protocol["identity"]["seeds"]
    def mean(condition: str, arm: str, metric: str, *, control: bool = False) -> float:
        values = [per_seed[str(seed)]["conditions"][condition][
            "controls" if control else "arms"][arm][metric] for seed in seeds]
        return sum(values) / len(values)
    packet = {condition["name"]: mean(condition["name"], "local_credit_packet", "accuracy")
              for condition in protocol["schedule"]["conditions"]}
    shortcut = mean("quarter_noise_immediate", "direct_anchor_shortcut", "accuracy")
    a = protocol["acceptance"]
    checks = {
        "clean_immediate": packet["clean_immediate"] >= a["clean_immediate_minimum_accuracy"],
        "quarter_noise": packet["quarter_noise_immediate"] >= a["quarter_noise_minimum_accuracy"],
        "quarter_noise_shortcut_gain": packet["quarter_noise_immediate"] - shortcut >= a["quarter_noise_minimum_shortcut_gain"],
        "half_noise_negative": packet["half_noise_immediate"] <= a["half_noise_maximum_accuracy"],
        "late_504_negative": packet["clean_late_504"] <= a["late_504_maximum_accuracy"],
        "after_deadline_negative": packet["clean_after_deadline_512"] <= a["after_deadline_maximum_accuracy"],
        "quarter_noise_map_reset": packet["quarter_noise_immediate"] - mean("quarter_noise_immediate", "B_local_map_reset", "accuracy", control=True) >= a["quarter_noise_minimum_map_reset_drop"],
        "quarter_noise_route_shuffle": packet["quarter_noise_immediate"] - mean("quarter_noise_immediate", "B_to_A_route_shuffle", "accuracy", control=True) >= a["quarter_noise_minimum_route_shuffle_drop"],
        "quarter_noise_sign_shuffle": packet["quarter_noise_immediate"] - mean("quarter_noise_immediate", "B_to_A_sign_shuffle", "accuracy", control=True) >= a["quarter_noise_minimum_sign_shuffle_drop"],
        "noise_monotone": packet["clean_immediate"] >= packet["quarter_noise_immediate"] >= packet["half_noise_immediate"],
        "late_monotone": packet["clean_late_480"] >= packet["clean_late_504"] >= packet["clean_after_deadline_512"],
    }
    all_results = [result for seed in per_seed.values() for stage in seed["conditions"].values()
                   for result in list(stage["arms"].values()) + list(stage["controls"].values())]
    b = protocol["resource_budgets"]
    harness = {
        "matched_forward_trace": all(
            len({result["forward_trace_sha256"] for result in list(stage["arms"].values()) + list(stage["controls"].values())}) == 1
            for seed in per_seed.values() for stage in seed["conditions"].values()),
        "capacity_matched_prediction": all(
            stage["arms"]["direct_anchor_shortcut"]["prediction_trace_sha256"]
            == stage["arms"]["capacity_matched_direct_shortcut"]["prediction_trace_sha256"]
            for seed in per_seed.values() for stage in seed["conditions"].values()),
        "capacity_state_allowance": all(
            stage["arms"]["capacity_matched_direct_shortcut"]["peak_state_bytes"]
            >= stage["arms"]["local_credit_packet"]["peak_state_bytes"]
            for seed in per_seed.values() for stage in seed["conditions"].values()),
        "A_feature_budget": max(result["A_features"] for result in all_results) <= b["maximum_A_features"],
        "B_feature_budget": max(result["B_features"] for result in all_results) <= b["maximum_B_features"],
        "pending_budget": max(result["maximum_pending_B_evidence_entries"] for result in all_results) <= b["maximum_pending_B_evidence_entries"],
        "state_budget": max(result["peak_state_bytes"] for result in all_results) <= b["maximum_total_state_bytes"],
        "packet_budget": max(result["maximum_packet_bytes"] for result in all_results) <= b["maximum_packet_bytes"],
        "eligibility_budget": max(result["maximum_A_eligibility_entries"] for result in all_results) <= b["maximum_A_eligibility_entries"],
        "exact_replay": all(
            replay_quarter_noise(protocol, seed)
            == per_seed[str(seed)]["conditions"]["quarter_noise_immediate"]["arms"]["local_credit_packet"]["prediction_trace_sha256"]
            for seed in seeds),
    }
    report = {"schema": "sara-local-credit-packet-noisy-late-B-evidence-development-v1",
              "protocol_sha256": PROTOCOL_SHA256,
              "mean_packet_accuracy_by_condition": packet,
              "mean_quarter_noise_shortcut_accuracy": shortcut,
              "checks": checks, "harness_checks": harness,
              "development_gate_passed": all(checks.values()),
              "harness_passed": all(harness.values()),
              "per_seed": per_seed, "heldout_consumed": False,
              "production_authorized": False}
    output = Path(ensure_parent_directory(OUTPUT))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "packet_accuracy": packet,
                      "shortcut_accuracy_quarter_noise": shortcut,
                      "checks": checks, "harness_checks": harness}, sort_keys=True))
    return 0 if report["development_gate_passed"] and report["harness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
