#!/usr/bin/env python3
"""Run the preregistered partial-B-supervision development experiment once."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.evaluation.local_credit_packet_multihop import (  # noqa: E402
    BLocalOutcome,
    CircuitA,
    CircuitB,
    GlobalOutcomeToB,
    _deep_size,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_weak_B_supervision_v1.json"
))
PROTOCOL_SHA256 = "cb48e2d6b246a2b82a0b94178353959c5a67f2fa763e14e33c91c8a272817887"
PARENT = ROOT / "src/sara_engine/evaluation/local_credit_packet_multihop.py"
HELDOUT = Path(workspace_path(
    "evaluation", "local_credit_packet_multihop_heldout_result_v1.json"
))
OUTPUT = Path(workspace_path(
    "evaluation", "local_credit_packet_weak_B_supervision_development_v1.json"
))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rank(namespace: str, seed: int, cue: int) -> str:
    return hashlib.sha256(f"{namespace}|{seed}|{cue}".encode()).hexdigest()


def balanced_map(namespace: str, seed: int, cues: list[int]) -> dict[int, int]:
    ordered = sorted(cues, key=lambda cue: rank(namespace, seed, cue))
    return {cue: int(index >= len(cues) // 2) for index, cue in enumerate(ordered)}


def materialize(protocol: dict, seed: int) -> dict:
    identity = protocol["identity"]
    namespace = identity["namespace"]
    A_cues = identity["A_cues"]
    B_cues = identity["B_cues"]
    A_map = balanced_map(namespace + "|A-map", seed, A_cues)
    B_map = balanced_map(namespace + "|B-map", seed, B_cues)
    ranked_B = sorted(B_cues, key=lambda cue: rank(namespace + "|visibility", seed, cue))
    calibration = [
        (B_cue, B_map[B_cue])
        for B_cue in B_cues for _ in range(16)
    ]
    training = []
    for repeat in range(2):
        for A_cue in A_cues:
            for B_cue in B_cues:
                for A_action in (0, 1):
                    for B_action in (0, 1):
                        training.append((A_cue, B_cue, A_action, B_action,
                                         int(B_action == (A_map[A_cue] ^ B_map[B_cue]))))
    development = [
        (A_cue, B_cue, A_map[A_cue], B_map[B_cue])
        for _ in range(2) for A_cue in A_cues for B_cue in B_cues
    ]
    return {"A_map": A_map, "B_map": B_map, "ranked_B": ranked_B,
            "calibration": calibration, "training": training, "development": development}


def run_arm(rows: dict, visible: set[int], arm: str, intervention: str = "none") -> dict:
    A = CircuitA(reserve=512 if arm == "capacity_matched_direct_shortcut" else 0)
    B = CircuitB()
    for B_cue, target in rows["calibration"]:
        if B_cue in visible:
            B.observe_local(BLocalOutcome(B_cue, target))
    peak_state = _deep_size((A.weights, A.eligibility, A.reserve, B.local_votes))
    forward = []
    packet_bytes = 0
    for index, (A_cue, B_cue, A_action, B_action, success) in enumerate(rows["training"]):
        source = index + 1
        A.record(source, A_cue, A_action)
        peak_state = max(peak_state, _deep_size((A.weights, A.eligibility, A.reserve, B.local_votes)))
        forward.append((index, A_action, B_action, success))
        if B_cue in visible:
            B.observe_local(BLocalOutcome(B_cue, rows["B_map"][B_cue]))
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
        peak_state = max(peak_state, _deep_size((A.weights, A.eligibility, A.reserve, B.local_votes)))
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
        "packet_bytes": packet_bytes,
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
    for path, expected in ((PARENT, protocol["parent_candidate_sha256"]),
                           (HELDOUT, protocol["parent_heldout_result_sha256"])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen parent changed: {path.name}")
    per_seed = {}
    for seed in protocol["identity"]["seeds"]:
        rows = materialize(protocol, seed)
        coverage = {}
        for count in protocol["coverage_cue_counts"]:
            visible = set(rows["ranked_B"][:count])
            arms = {arm: run_arm(rows, visible, arm) for arm in protocol["arms"]}
            controls = ({name: run_arm(rows, visible, "local_credit_packet", name)
                         for name in protocol["interventions_at_six_cues"]}
                        if count == 6 else {})
            coverage[str(count)] = {"visible_B_cues": sorted(visible),
                                    "arms": arms, "controls": controls}
        per_seed[str(seed)] = {"coverage": coverage,
                              "materialization_sha256": digest(rows)}
    def mean(count: int, arm: str, metric: str, *, control: bool = False) -> float:
        values = [per_seed[str(seed)]["coverage"][str(count)][
            "controls" if control else "arms"][arm][metric]
            for seed in protocol["identity"]["seeds"]]
        return sum(values) / len(values)
    packet = {str(count): mean(count, "local_credit_packet", "accuracy")
              for count in protocol["coverage_cue_counts"]}
    shortcut = mean(6, "direct_anchor_shortcut", "accuracy")
    acceptance = protocol["acceptance"]
    checks = {
        "six_cue_accuracy": packet["6"] >= acceptance["six_cue_minimum_packet_accuracy"],
        "six_cue_shortcut_gain": packet["6"] - shortcut >= acceptance["six_cue_minimum_gain_over_shortcut"],
        "six_cue_B_accuracy": mean(6, "local_credit_packet", "B_local_accuracy") >= acceptance["six_cue_minimum_B_local_accuracy"],
        "zero_cue_negative": packet["0"] <= acceptance["zero_cue_maximum_packet_accuracy"],
        "six_cue_map_reset": packet["6"] - mean(6, "B_local_map_reset", "accuracy", control=True) >= acceptance["six_cue_minimum_map_reset_drop"],
        "six_cue_sign_shuffle": packet["6"] - mean(6, "B_to_A_sign_shuffle", "accuracy", control=True) >= acceptance["six_cue_minimum_sign_shuffle_drop"],
        "six_cue_route_shuffle": packet["6"] - mean(6, "B_to_A_route_shuffle", "accuracy", control=True) >= acceptance["six_cue_minimum_route_shuffle_drop"],
        "five_seed_gains": all(
            per_seed[str(seed)]["coverage"]["6"]["arms"]["local_credit_packet"]["accuracy"]
            > per_seed[str(seed)]["coverage"]["6"]["arms"]["direct_anchor_shortcut"]["accuracy"]
            for seed in protocol["identity"]["seeds"]),
        "monotone": packet["8"] >= packet["6"] >= packet["4"] >= packet["0"],
    }
    all_results = [result for seed in per_seed.values() for coverage in seed["coverage"].values()
                   for result in list(coverage["arms"].values()) + list(coverage["controls"].values())]
    budgets = protocol["resource_budgets"]
    harness = {
        "forward_trace_matched": all(
            len({result["forward_trace_sha256"] for result in list(coverage["arms"].values()) + list(coverage["controls"].values())}) == 1
            for seed in per_seed.values() for coverage in seed["coverage"].values()),
        "capacity_matched_prediction": all(
            coverage["arms"]["direct_anchor_shortcut"]["prediction_trace_sha256"]
            == coverage["arms"]["capacity_matched_direct_shortcut"]["prediction_trace_sha256"]
            for seed in per_seed.values() for coverage in seed["coverage"].values()),
        "capacity_state_allowance": all(
            coverage["arms"]["capacity_matched_direct_shortcut"]["peak_state_bytes"]
            >= coverage["arms"]["local_credit_packet"]["peak_state_bytes"]
            for seed in per_seed.values() for coverage in seed["coverage"].values()),
        "A_feature_budget": max(row["A_features"] for row in all_results) <= budgets["maximum_A_features"],
        "B_feature_budget": max(row["B_features"] for row in all_results) <= budgets["maximum_B_features"],
        "state_budget": max(row["peak_state_bytes"] for row in all_results) <= budgets["maximum_total_state_bytes"],
        "packet_budget": max(row["packet_bytes"] for row in all_results) <= budgets["maximum_packet_bytes"],
        "eligibility_budget": max(row["maximum_A_eligibility_entries"] for row in all_results) <= 1,
        "exact_replay": all(
            run_arm(materialize(protocol, seed), set(materialize(protocol, seed)["ranked_B"][:6]), "local_credit_packet")["prediction_trace_sha256"]
            == per_seed[str(seed)]["coverage"]["6"]["arms"]["local_credit_packet"]["prediction_trace_sha256"]
            for seed in protocol["identity"]["seeds"]),
    }
    report = {"schema": "sara-local-credit-packet-weak-B-supervision-development-v1",
              "protocol_sha256": PROTOCOL_SHA256, "mean_packet_accuracy_by_visible_B_cues": packet,
              "mean_six_cue_shortcut_accuracy": shortcut, "checks": checks,
              "harness_checks": harness, "development_gate_passed": all(checks.values()),
              "harness_passed": all(harness.values()), "per_seed": per_seed,
              "heldout_consumed": False, "production_authorized": False}
    output = Path(ensure_parent_directory(OUTPUT))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "packet_accuracy": packet,
                      "shortcut_accuracy_six_cues": shortcut, "checks": checks,
                      "harness_checks": harness}, sort_keys=True))
    return 0 if report["development_gate_passed"] and report["harness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
