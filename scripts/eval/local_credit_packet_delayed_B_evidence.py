#!/usr/bin/env python3
"""Evaluate frozen episode-delayed B-local evidence without retroactive replay."""
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
    "benchmark_fixtures", "local_credit_packet_delayed_B_evidence_v1.json"
))
PROTOCOL_SHA256 = "78ee896b9d9f7f74b5840a3c0e9a1464c6f1089de5e4ef4f51a607a96e40115f"
PARENT = ROOT / "src/sara_engine/evaluation/local_credit_packet_multihop.py"
MATERIALIZER = ROOT / "scripts/eval/local_credit_packet_weak_B_supervision.py"
PARENT_REPORT = Path(workspace_path(
    "evaluation", "local_credit_packet_weak_B_supervision_development_v1.json"
))
OUTPUT = Path(workspace_path(
    "evaluation", "local_credit_packet_delayed_B_evidence_development_v1.json"
))


def run_arm(rows: dict, delay: int, arm: str, intervention: str = "none") -> dict:
    A = CircuitA(reserve=512 if arm == "capacity_matched_direct_shortcut" else 0)
    B = CircuitB()
    pending: deque[tuple[int, int, int]] = deque()
    maximum_pending = 0
    packet_bytes = 0
    forward = []
    peak_state = _deep_size((A.weights, A.eligibility, A.reserve, B.local_votes, pending))
    for index, (A_cue, B_cue, A_action, B_action, success) in enumerate(rows["training"]):
        while pending and pending[0][0] <= index:
            _, cue, target = pending.popleft()
            B.observe_local(BLocalOutcome(cue, target))
        source = index + 1
        A.record(source, A_cue, A_action)
        forward.append((index, A_action, B_action, success))
        if delay == 0:
            B.observe_local(BLocalOutcome(B_cue, rows["B_map"][B_cue]))
        else:
            pending.append((index + delay, B_cue, rows["B_map"][B_cue]))
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
            packet_bytes = max(packet_bytes, len(packet.to_bytes()))
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
        "B_local_updates": B.local_updates,
        "A_updates": A.updates,
        "pending_B_evidence_entries_at_scoring": len(pending),
        "maximum_pending_B_evidence_entries": maximum_pending,
        "maximum_packet_bytes": packet_bytes,
        "peak_state_bytes": peak_state,
        "A_features": len(A.weights),
        "B_features": len(B.local_votes),
        "maximum_A_eligibility_entries": A.maximum_eligibility_entries,
        "forward_trace_sha256": digest(forward),
        "prediction_trace_sha256": digest(predictions),
    }


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Development result already exists; refusing to rerun")
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Preregistered protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    for path, expected in (
        (PARENT, protocol["parent_candidate_sha256"]),
        (MATERIALIZER, protocol["parent_materializer_sha256"]),
        (PARENT_REPORT, protocol["parent_partial_coverage_result_sha256"]),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen parent changed: {path.name}")
    per_seed = {}
    for seed in protocol["identity"]["seeds"]:
        rows = materialize(protocol, seed)
        delays = {}
        for delay in protocol["schedule"]["B_evidence_delays_in_episodes"]:
            arms = {arm: run_arm(rows, delay, arm) for arm in protocol["arms"]}
            controls = ({name: run_arm(rows, delay, "local_credit_packet", name)
                         for name in protocol["interventions_at_delay_64"]}
                        if delay == 64 else {})
            delays[str(delay)] = {"arms": arms, "controls": controls}
        per_seed[str(seed)] = {"delays": delays, "materialization_sha256": digest(rows)}
    seeds = protocol["identity"]["seeds"]
    def mean(delay: int, arm: str, metric: str, *, control: bool = False) -> float:
        values = [per_seed[str(seed)]["delays"][str(delay)][
            "controls" if control else "arms"][arm][metric] for seed in seeds]
        return sum(values) / len(values)
    packet = {str(delay): mean(delay, "local_credit_packet", "accuracy")
              for delay in protocol["schedule"]["B_evidence_delays_in_episodes"]}
    shortcut = mean(64, "direct_anchor_shortcut", "accuracy")
    acceptance = protocol["acceptance"]
    checks = {
        "zero_delay_accuracy": packet["0"] >= acceptance["zero_delay_minimum_packet_accuracy"],
        "delay_64_accuracy": packet["64"] >= acceptance["delay_64_minimum_packet_accuracy"],
        "delay_64_shortcut_gain": packet["64"] - shortcut >= acceptance["delay_64_minimum_shortcut_gain"],
        "delay_512_negative": packet["512"] <= acceptance["delay_512_maximum_packet_accuracy"],
        "delay_64_map_reset": packet["64"] - mean(64, "B_local_map_reset", "accuracy", control=True) >= acceptance["delay_64_minimum_map_reset_drop"],
        "delay_64_route_shuffle": packet["64"] - mean(64, "B_to_A_route_shuffle", "accuracy", control=True) >= acceptance["delay_64_minimum_route_shuffle_drop"],
        "delay_64_sign_shuffle": packet["64"] - mean(64, "B_to_A_sign_shuffle", "accuracy", control=True) >= acceptance["delay_64_minimum_sign_shuffle_drop"],
        "five_seed_gains": all(
            per_seed[str(seed)]["delays"]["64"]["arms"]["local_credit_packet"]["accuracy"]
            > per_seed[str(seed)]["delays"]["64"]["arms"]["direct_anchor_shortcut"]["accuracy"]
            for seed in seeds),
        "monotone": packet["0"] >= packet["16"] >= packet["64"] >= packet["256"] >= packet["512"],
    }
    all_results = [result for seed in per_seed.values() for stage in seed["delays"].values()
                   for result in list(stage["arms"].values()) + list(stage["controls"].values())]
    budgets = protocol["resource_budgets"]
    harness = {
        "matched_forward_trace": all(
            len({result["forward_trace_sha256"] for result in list(stage["arms"].values()) + list(stage["controls"].values())}) == 1
            for seed in per_seed.values() for stage in seed["delays"].values()),
        "capacity_matched_prediction": all(
            stage["arms"]["direct_anchor_shortcut"]["prediction_trace_sha256"]
            == stage["arms"]["capacity_matched_direct_shortcut"]["prediction_trace_sha256"]
            for seed in per_seed.values() for stage in seed["delays"].values()),
        "capacity_state_allowance": all(
            stage["arms"]["capacity_matched_direct_shortcut"]["peak_state_bytes"]
            >= stage["arms"]["local_credit_packet"]["peak_state_bytes"]
            for seed in per_seed.values() for stage in seed["delays"].values()),
        "A_feature_budget": max(row["A_features"] for row in all_results) <= budgets["maximum_A_features"],
        "B_feature_budget": max(row["B_features"] for row in all_results) <= budgets["maximum_B_features"],
        "pending_budget": max(row["maximum_pending_B_evidence_entries"] for row in all_results) <= budgets["maximum_pending_B_evidence_entries"],
        "state_budget": max(row["peak_state_bytes"] for row in all_results) <= budgets["maximum_total_state_bytes"],
        "packet_budget": max(row["maximum_packet_bytes"] for row in all_results) <= budgets["maximum_packet_bytes"],
        "eligibility_budget": max(row["maximum_A_eligibility_entries"] for row in all_results) <= budgets["maximum_A_eligibility_entries"],
        "exact_replay": all(
            run_arm(materialize(protocol, seed), 64, "local_credit_packet")["prediction_trace_sha256"]
            == per_seed[str(seed)]["delays"]["64"]["arms"]["local_credit_packet"]["prediction_trace_sha256"]
            for seed in seeds),
    }
    report = {"schema": "sara-local-credit-packet-delayed-B-evidence-development-v1",
              "protocol_sha256": PROTOCOL_SHA256,
              "mean_packet_accuracy_by_delay": packet,
              "mean_delay_64_shortcut_accuracy": shortcut,
              "checks": checks, "harness_checks": harness,
              "development_gate_passed": all(checks.values()),
              "harness_passed": all(harness.values()),
              "per_seed": per_seed, "heldout_consumed": False,
              "production_authorized": False}
    output = Path(ensure_parent_directory(OUTPUT))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "packet_accuracy": packet,
                      "shortcut_accuracy_delay_64": shortcut,
                      "checks": checks, "harness_checks": harness}, sort_keys=True))
    return 0 if report["development_gate_passed"] and report["harness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
