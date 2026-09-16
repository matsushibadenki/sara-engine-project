#!/usr/bin/env python3
"""Execute the frozen prior-error repair held-out gate exactly once."""
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

PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_prior_error_repair_heldout_v1.json"
EXECUTION = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_prior_error_repair_heldout_execution_v1.json"
CANDIDATE = ROOT / "scripts/eval/local_credit_packet_prior_error_repair.py"
MATERIALIZER = ROOT / "scripts/eval/local_credit_packet_prior_error_repair_heldout_materialize.py"
ROWS = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_prior_error_repair_heldout_rows_v1.jsonl"
AUDIT = Path(workspace_path("evaluation", "local_credit_packet_prior_error_repair_heldout_audit_v1.json"))
DEVELOPMENT = Path(workspace_path("evaluation", "local_credit_packet_prior_error_repair_development_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "local_credit_packet_prior_error_repair_heldout_result_v1.json"))
EXECUTION_SHA256 = "c8149956443c5a47f11a87ee4c88f430a4750415d5e1517fa8b0799081c2bc6b"


def _verify(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Frozen input changed: {path.name}")


def _adapt(seed_rows: list[dict], protocol: dict) -> tuple[dict, list[tuple]]:
    train_rows = [row for row in seed_rows if row["phase"] == "training"]
    test_rows = [row for row in seed_rows if row["phase"] == "test"]
    if len(train_rows) != 512 or len(test_rows) != 128:
        raise ValueError("Invalid held-out split")
    A_map: dict[int, int] = {}
    B_map: dict[int, int] = {}
    for row in train_rows:
        A_cue, B_cue = int(row["A_cue"]), int(row["B_cue"])
        A_target, B_target = int(row["A_target"]), int(row["B_local_target"])
        if A_cue in A_map and A_map[A_cue] != A_target:
            raise ValueError("Inconsistent A training target")
        if B_cue in B_map and B_map[B_cue] != B_target:
            raise ValueError("Inconsistent B training target")
        A_map[A_cue] = A_target
        B_map[B_cue] = B_target
    identity = protocol["identity"]
    if set(A_map) != set(identity["A_cues"]) or set(B_map) != set(identity["B_cues"]):
        raise ValueError("Incomplete private map")
    training = [
        (int(row["A_cue"]), int(row["B_cue"]), int(row["forced_A_action"]),
         int(row["forced_B_action"]), int(row["global_success"]))
        for row in train_rows
    ]
    test = [
        (int(row["A_cue"]), int(row["B_cue"]), int(row["A_target"]),
         int(row["B_local_target"])) for row in test_rows
    ]
    return {"A_map": A_map, "B_map": B_map, "development": test}, training


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("One-shot result already exists; refusing to rerun")
    _verify(EXECUTION, EXECUTION_SHA256)
    execution = json.loads(EXECUTION.read_text())
    for path, key in (
        (PROTOCOL, "protocol_sha256"),
        (CANDIDATE, "candidate_source_sha256"),
        (MATERIALIZER, "materializer_source_sha256"),
        (ROWS, "rows_sha256"),
        (AUDIT, "pre_execution_audit_sha256"),
    ):
        _verify(path, execution[key])
    protocol = json.loads(PROTOCOL.read_text())
    _verify(DEVELOPMENT, protocol["parent_development_result_sha256"])
    audit = json.loads(AUDIT.read_text())
    if audit["passed"] is not True or audit["heldout_consumed"] is not False:
        raise ValueError("Pre-execution audit failed")
    if audit["protocol_sha256"] != execution["protocol_sha256"] or audit["rows_sha256"] != execution["rows_sha256"]:
        raise ValueError("Audit binding mismatch")

    from scripts.eval.local_credit_packet_prior_error_repair import (  # noqa: E402
        ARMS, CONTROLS, run_arm,
    )

    if list(ARMS) != protocol["evaluation"]["arms"] or list(CONTROLS) != protocol["evaluation"]["controls"]:
        raise ValueError("Candidate arm contract changed")
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    seeds = protocol["identity"]["seeds"]
    per_seed = {}
    for seed in seeds:
        adapted, training = _adapt([row for row in rows if row["seed"] == seed], protocol)
        arms = {arm: run_arm(adapted, training, arm) for arm in ARMS}
        controls = {name: run_arm(
            adapted, training,
            "capacity_matched_direct_correction" if name == "capacity_matched_direct_correction"
            else "correction_packet",
            "none" if name == "capacity_matched_direct_correction" else name,
        ) for name in CONTROLS}
        per_seed[str(seed)] = {"arms": arms, "controls": controls}
    def mean(arm: str, metric: str, *, control: bool = False) -> float:
        values = [per_seed[str(seed)]["controls" if control else "arms"][arm][metric]
                  for seed in seeds]
        return sum(values) / len(values)
    summary = {arm: mean(arm, "accuracy") for arm in ARMS}
    control_summary = {name: mean(name, "accuracy", control=True) for name in CONTROLS}
    correction = summary["correction_packet"]
    a = protocol["acceptance"]
    checks = {
        "correction_accuracy": correction >= a["minimum_correction_accuracy"],
        "wrong_credit_negative": summary["wrong_credit_only"] <= a["maximum_wrong_credit_accuracy"],
        "simple_replay_negative": summary["simple_replay"] <= a["maximum_simple_replay_accuracy"],
        "gain_over_simple": correction - summary["simple_replay"] >= a["minimum_gain_over_simple_replay"],
        "direct_outcome_negative": summary["direct_outcome_correction"] <= a["maximum_direct_outcome_accuracy"],
        "oracle": summary["oracle_control"] >= a["minimum_oracle_accuracy"],
        "B_map_reset": correction - control_summary["B_local_map_reset_on_replay"] >= a["minimum_B_map_reset_drop"],
        "route_shuffle": correction - control_summary["B_to_A_route_shuffle_on_replay"] >= a["minimum_route_shuffle_drop"],
        "sign_shuffle": correction - control_summary["B_to_A_sign_shuffle_on_replay"] >= a["minimum_sign_shuffle_drop"],
        "anchor_erasure": correction - control_summary["A_anchor_erased_before_replay"] >= a["minimum_anchor_erasure_drop"],
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
        "replay_lookup_budget": max(result["replay_lookups"] for result in all_results) <= b["maximum_replay_lookups"],
        "replay_packet_budget": max(result["replay_packets"] for result in all_results) <= b["maximum_replay_packets"],
        "packet_bytes": max(result["maximum_packet_bytes"] for result in all_results) <= b["maximum_packet_bytes"],
        "eligibility_budget": max(result["maximum_A_eligibility_entries"] for result in all_results) <= b["maximum_A_eligibility_entries"],
        "A_feature_budget": max(result["A_features"] for result in all_results) <= b["maximum_A_features"],
        "B_feature_budget": max(result["B_features"] for result in all_results) <= b["maximum_B_features"],
        "state_budget": max(result["peak_state_bytes"] for result in all_results) <= b["maximum_total_state_bytes"],
        "exact_replay": all(
            run_arm(*_adapt([row for row in rows if row["seed"] == seed], protocol),
                    "correction_packet")["prediction_trace_sha256"]
            == per_seed[str(seed)]["arms"]["correction_packet"]["prediction_trace_sha256"]
            for seed in seeds),
    }
    report = {
        "schema": "sara-local-credit-packet-prior-error-repair-heldout-result-v1",
        "protocol_sha256": execution["protocol_sha256"],
        "execution_sha256": EXECUTION_SHA256,
        "rows_sha256": execution["rows_sha256"],
        "mean_accuracy": summary,
        "control_accuracy": control_summary,
        "heldout_checks": checks,
        "harness_checks": harness,
        "heldout_gate_passed": all(checks.values()),
        "harness_passed": all(harness.values()),
        "per_seed": per_seed,
        "heldout_consumed": True,
        "production_authorized": False,
    }
    output = Path(ensure_parent_directory(OUTPUT))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "mean_accuracy": summary,
                      "control_accuracy": control_summary,
                      "heldout_checks": checks, "harness_checks": harness}, sort_keys=True))
    return 0 if report["heldout_gate_passed"] and report["harness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
