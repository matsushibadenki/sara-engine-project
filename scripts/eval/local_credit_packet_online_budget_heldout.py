#!/usr/bin/env python3
"""Execute the independently materialized online-budget held-out gate once."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory, workspace_path,
)

PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_online_budget_heldout_v1.json"
EXECUTION = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_online_budget_heldout_execution_v1.json"
GENERATOR = ROOT / "scripts/eval/local_credit_packet_online_budget_heldout_materialize.py"
ROWS = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_online_budget_heldout_rows_v1.jsonl"
AUDIT = Path(workspace_path("evaluation", "local_credit_packet_online_budget_heldout_audit_v1.json"))
CANDIDATE = ROOT / "scripts/eval/local_credit_packet_online_budget_curve.py"
DEVELOPMENT = Path(workspace_path("evaluation", "local_credit_packet_online_budget_curve_development_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "local_credit_packet_online_budget_heldout_result_v1.json"))
EXECUTION_SHA256 = "30672fea413f27b857285a559d48b91c78e2a67610c164a4a2727afae11563f4"


def _verify(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Frozen input changed: {path.name}")


def _adapt(seed_rows: list[dict], protocol: dict) -> tuple[dict, list[tuple]]:
    training_rows = [row for row in seed_rows if row["phase"] == "training"]
    test_rows = [row for row in seed_rows if row["phase"] == "test"]
    if len(training_rows) != 512 or len(test_rows) != 128:
        raise ValueError("Invalid per-seed split")
    A_map: dict[int, int] = {}
    B_map: dict[int, int] = {}
    for row in training_rows:
        A_cue, B_cue = int(row["A_cue"]), int(row["B_cue"])
        A_target, B_target = int(row["A_target"]), int(row["B_local_target"])
        if A_cue in A_map and A_map[A_cue] != A_target:
            raise ValueError("Inconsistent training A map")
        if B_cue in B_map and B_map[B_cue] != B_target:
            raise ValueError("Inconsistent training B map")
        A_map[A_cue] = A_target
        B_map[B_cue] = B_target
    identity = protocol["identity"]
    if set(A_map) != set(identity["A_cues"]) or set(B_map) != set(identity["B_cues"]):
        raise ValueError("Incomplete training private maps")
    training = [
        (int(row["A_cue"]), int(row["B_cue"]), int(row["forced_A_action"]),
         int(row["forced_B_action"]), int(row["global_success"]))
        for row in training_rows
    ]
    test = [
        (int(row["A_cue"]), int(row["B_cue"]), int(row["A_target"]),
         int(row["B_local_target"]))
        for row in test_rows
    ]
    return {"A_map": A_map, "B_map": B_map, "development": test}, training


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("One-shot held-out result already exists; refusing to rerun")
    _verify(EXECUTION, EXECUTION_SHA256)
    execution = json.loads(EXECUTION.read_text())
    for path, key in (
        (PROTOCOL, "protocol_sha256"),
        (GENERATOR, "generator_source_sha256"),
        (ROWS, "rows_sha256"),
        (AUDIT, "pre_execution_audit_sha256"),
        (CANDIDATE, "candidate_source_sha256"),
    ):
        _verify(path, execution[key])
    protocol = json.loads(PROTOCOL.read_text())
    _verify(DEVELOPMENT, protocol["parent_development_result_sha256"])
    audit = json.loads(AUDIT.read_text())
    if audit["passed"] is not True or audit["heldout_consumed"] is not False:
        raise ValueError("Pre-execution audit did not pass")
    if audit["protocol_sha256"] != execution["protocol_sha256"] or audit["rows_sha256"] != execution["rows_sha256"]:
        raise ValueError("Pre-execution audit binding mismatch")

    from scripts.eval.local_credit_packet_online_budget_curve import (  # noqa: E402
        CONTROLS, ARMS, run_arm,
    )

    if list(ARMS) != protocol["evaluation"]["arms"] or list(CONTROLS) != execution["comparison"]["zero_budget_controls"]:
        raise ValueError("Candidate arm contract changed")
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    seeds = protocol["identity"]["seeds"]
    per_seed = {}
    for seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        adapted, training = _adapt(seed_rows, protocol)
        budgets = {}
        for budget in protocol["evaluation"]["online_budgets"]:
            arms = {arm: run_arm(adapted, training, budget, arm) for arm in ARMS}
            controls = ({name: run_arm(
                adapted, training, budget,
                "capacity_matched_direct_shortcut" if name == "capacity_matched_direct_shortcut"
                else "targeted_replay_packet",
                "none" if name == "capacity_matched_direct_shortcut" else name,
            ) for name in CONTROLS} if budget == 0 else {})
            budgets[str(budget)] = {"arms": arms, "controls": controls}
        per_seed[str(seed)] = {"budgets": budgets}
    def mean(budget: int, arm: str, metric: str, *, control: bool = False) -> float:
        values = [per_seed[str(seed)]["budgets"][str(budget)][
            "controls" if control else "arms"][arm][metric] for seed in seeds]
        return sum(values) / len(values)
    summary = {str(budget): {arm: mean(budget, arm, "accuracy") for arm in ARMS}
               for budget in protocol["evaluation"]["online_budgets"]}
    control_summary = {name: mean(0, name, "accuracy", control=True) for name in CONTROLS}
    a = protocol["acceptance"]
    packet0 = summary["0"]["targeted_replay_packet"]
    packet2 = summary["2"]["targeted_replay_packet"]
    checks = {
        "zero_packet_accuracy": packet0 >= a["zero_budget_minimum_packet_accuracy"],
        "zero_online_gain": packet0 - summary["0"]["online_only_packet"] >= a["zero_budget_minimum_gain_over_online_only"],
        "zero_direct_shortcut": summary["0"]["targeted_replay_direct_shortcut"] <= a["zero_budget_maximum_direct_shortcut_accuracy"],
        "two_packet_accuracy": packet2 >= a["two_budget_minimum_packet_accuracy"],
        "two_online_gain": packet2 - summary["2"]["online_only_packet"] >= a["two_budget_minimum_gain_over_online_only"],
        "B_map_reset": packet0 - control_summary["B_local_map_reset_on_replay"] >= a["zero_budget_minimum_map_reset_drop"],
        "route_shuffle": packet0 - control_summary["B_to_A_route_shuffle_on_replay"] >= a["zero_budget_minimum_route_shuffle_drop"],
        "sign_shuffle": packet0 - control_summary["B_to_A_sign_shuffle_on_replay"] >= a["zero_budget_minimum_sign_shuffle_drop"],
        "anchor_erasure": packet0 - control_summary["A_anchor_erased_before_replay"] >= a["zero_budget_minimum_anchor_erasure_drop"],
        "five_seed_gains": all(
            per_seed[str(seed)]["budgets"][str(budget)]["arms"]["targeted_replay_packet"]["accuracy"]
            > per_seed[str(seed)]["budgets"][str(budget)]["arms"]["online_only_packet"]["accuracy"]
            for seed in seeds for budget in (0, 2)),
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
        "eligibility_budget": max(result["maximum_A_eligibility_entries"] for result in all_results) <= b["maximum_A_eligibility_entries"],
        "A_feature_budget": max(result["A_features"] for result in all_results) <= b["maximum_A_features"],
        "B_feature_budget": max(result["B_features"] for result in all_results) <= b["maximum_B_features"],
        "state_budget": max(result["peak_state_bytes"] for result in all_results) <= b["maximum_total_state_bytes"],
        "exact_replay": all(
            run_arm(*_adapt([row for row in rows if row["seed"] == seed], protocol),
                    0, "targeted_replay_packet")["prediction_trace_sha256"]
            == per_seed[str(seed)]["budgets"]["0"]["arms"]["targeted_replay_packet"]["prediction_trace_sha256"]
            for seed in seeds),
    }
    report = {"schema": "sara-local-credit-packet-online-budget-heldout-result-v1",
              "protocol_sha256": execution["protocol_sha256"],
              "execution_sha256": EXECUTION_SHA256,
              "rows_sha256": execution["rows_sha256"],
              "accuracy_by_online_budget": summary,
              "zero_budget_control_accuracy": control_summary,
              "heldout_checks": checks, "harness_checks": harness,
              "heldout_gate_passed": all(checks.values()),
              "harness_passed": all(harness.values()),
              "per_seed": per_seed, "heldout_consumed": True,
              "production_authorized": False}
    output = Path(ensure_parent_directory(OUTPUT))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "accuracy_by_online_budget": summary,
                      "heldout_checks": checks, "harness_checks": harness}, sort_keys=True))
    return 0 if report["heldout_gate_passed"] and report["harness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
