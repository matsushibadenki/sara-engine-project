#!/usr/bin/env python3
"""Score the frozen mixed-error revision held-out set exactly once."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory, processed_data_path, workspace_path,
)

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_mixed_error_revision_heldout_v1.json"
))
EXECUTION = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_mixed_error_revision_heldout_execution_v1.json"
))
CANDIDATE = ROOT / "scripts/eval/local_credit_packet_mixed_error_revision.py"
MATERIALIZER = ROOT / "scripts/eval/local_credit_packet_mixed_error_revision_heldout_materialize.py"
ROWS = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_mixed_error_revision_heldout_rows_v1.jsonl"
))
AUDIT = Path(workspace_path(
    "evaluation", "local_credit_packet_mixed_error_revision_heldout_audit_v1.json"
))
DEVELOPMENT = Path(workspace_path(
    "evaluation", "local_credit_packet_mixed_error_revision_development_v1.json"
))
OUTPUT = Path(workspace_path(
    "evaluation", "local_credit_packet_mixed_error_revision_heldout_result_v1.json"
))
EXECUTION_SHA256 = "e953312ed88cd3fd121e2876031d602606824be7f4c605c29728e37b7efcd9f5"


def _verify(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Frozen input changed: {path.name}")


def _adapt(seed_rows: list[dict], protocol: dict) -> tuple[dict, list[tuple]]:
    train = [row for row in seed_rows if row["phase"] == "training"]
    test = [row for row in seed_rows if row["phase"] == "test"]
    if len(train) != 512 or len(test) != 128:
        raise ValueError("Invalid held-out split")
    A_map = {row["A_cue"]: row["A_target"] for row in train}
    B_map = {row["B_cue"]: row["B_local_target"] for row in train}
    identity = protocol["identity"]
    if set(A_map) != set(identity["A_cues"]) or set(B_map) != set(identity["B_cues"]):
        raise ValueError("Incomplete private map")
    if any(row["A_target"] != A_map[row["A_cue"]]
           or row["B_local_target"] != B_map[row["B_cue"]] for row in train + test):
        raise ValueError("Inconsistent private map")
    training = [(row["A_cue"], row["B_cue"], row["forced_A_action"],
                 row["forced_B_action"], row["global_success"]) for row in train]
    testing = [(row["A_cue"], row["B_cue"], row["A_target"],
                row["B_local_target"]) for row in test]
    return {"A_map": A_map, "B_map": B_map, "development": testing}, training


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("One-shot result already exists; refusing to rerun")
    _verify(EXECUTION, EXECUTION_SHA256)
    execution = json.loads(EXECUTION.read_text())
    for path, key in ((PROTOCOL, "protocol_sha256"),
                      (CANDIDATE, "candidate_source_sha256"),
                      (MATERIALIZER, "materializer_source_sha256"),
                      (ROWS, "rows_sha256"),
                      (AUDIT, "pre_execution_audit_sha256")):
        _verify(path, execution[key])
    protocol = json.loads(PROTOCOL.read_text())
    _verify(DEVELOPMENT, protocol["parent_development_result_sha256"])
    audit = json.loads(AUDIT.read_text())
    if audit["passed"] is not True or audit["heldout_consumed"] is not False:
        raise ValueError("Pre-execution audit failed")
    if audit["protocol_sha256"] != execution["protocol_sha256"] or audit["rows_sha256"] != execution["rows_sha256"]:
        raise ValueError("Audit binding mismatch")

    from scripts.eval.local_credit_packet_mixed_error_revision import (  # noqa: E402
        ARMS, CONDITIONS, CONTROLS, local_maps, run_arm,
    )

    if list(ARMS) != protocol["evaluation"]["arms"] or list(CONTROLS) != protocol["evaluation"]["full_revision_controls"]:
        raise ValueError("Candidate arm contract changed")
    if list(CONDITIONS) != list(protocol["schedule"]["late_B_revision"]):
        raise ValueError("Candidate condition contract changed")
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    seeds = protocol["identity"]["seeds"]
    per_seed = {}
    for seed in seeds:
        adapted, training = _adapt([row for row in rows if row["seed"] == seed], protocol)
        conditions = {}
        for condition in CONDITIONS:
            initial, revised, selection = local_maps(protocol, seed, adapted["B_map"], condition)
            arms = {arm: run_arm(adapted, training, initial, revised, arm) for arm in ARMS}
            controls = ({name: run_arm(
                adapted, training, initial, revised,
                "capacity_matched_direct_correction" if name == "capacity_matched_direct_correction"
                else "correction_packet",
                "none" if name == "capacity_matched_direct_correction" else name,
            ) for name in CONTROLS} if condition == "full" else {})
            conditions[condition] = {"selection": selection, "arms": arms,
                                     "controls": controls}
        per_seed[str(seed)] = {"conditions": conditions}
    def mean(condition: str, arm: str, metric: str, *, control: bool = False) -> float:
        values = [per_seed[str(seed)]["conditions"][condition][
            "controls" if control else "arms"][arm][metric] for seed in seeds]
        return sum(values) / len(values)
    summary = {condition: {arm: mean(condition, arm, "accuracy") for arm in ARMS}
               for condition in CONDITIONS}
    controls = {name: mean("full", name, "accuracy", control=True) for name in CONTROLS}
    correction = summary["full"]["correction_packet"]
    a = protocol["acceptance"]
    checks = {
        "full_correction": correction >= a["full_minimum_correction_accuracy"],
        "prior_gain": correction - summary["full"]["prior_credit_only"] >= a["full_minimum_gain_over_prior_credit"],
        "simple_gain": correction - summary["full"]["simple_replay"] >= a["full_minimum_gain_over_simple_replay"],
        "direct_outcome": summary["full"]["direct_outcome_correction"] <= a["full_maximum_direct_outcome_accuracy"],
        "oracle": summary["full"]["oracle_control"] >= a["full_minimum_oracle_accuracy"],
        "B_map_reset": correction - controls["B_local_map_reset_on_replay"] >= a["full_minimum_B_map_reset_drop"],
        "route_shuffle": correction - controls["B_to_A_route_shuffle_on_replay"] >= a["full_minimum_route_shuffle_drop"],
        "sign_shuffle": correction - controls["B_to_A_sign_shuffle_on_replay"] >= a["full_minimum_sign_shuffle_drop"],
        "anchor_erasure": correction - controls["A_anchor_erased_before_replay"] >= a["full_minimum_anchor_erasure_drop"],
        "full_at_least_imperfect": correction >= summary["partial_six"]["correction_packet"]
        and correction >= summary["one_corrupt"]["correction_packet"],
        "five_seed_simple_gains": all(
            per_seed[str(seed)]["conditions"]["full"]["arms"]["correction_packet"]["accuracy"]
            > per_seed[str(seed)]["conditions"]["full"]["arms"]["simple_replay"]["accuracy"]
            for seed in seeds),
    }
    all_results = [result for seed in per_seed.values() for stage in seed["conditions"].values()
                   for result in list(stage["arms"].values()) + list(stage["controls"].values())]
    b = protocol["resource_budgets"]
    harness = {
        "matched_forward_trace": all(
            len({result["forward_trace_sha256"] for result in list(stage["arms"].values())
                 + list(stage["controls"].values())}) == 1
            for seed in per_seed.values() for stage in seed["conditions"].values()),
        "sixteen_early_updates": all(result["early_packets"] == 16 for result in all_results),
        "B_revision_updates": all(result["B_local_updates"] == 16 for result in all_results),
        "full_B_local_accuracy": all(
            result["B_local_accuracy"] == 1.0 for seed in per_seed.values()
            for result in list(seed["conditions"]["full"]["arms"].values())
            + list(seed["conditions"]["full"]["controls"].values())),
        "independent_selection_and_oracle": all(
            stage["selection"] == {key: value for key, value in (
                ("initially_wrong_B_cues", audit["independent_oracle"]["per_seed"][str(seed)]["initially_wrong_B_cues"]),
                ("visible_revision_B_cues", audit["independent_oracle"]["per_seed"][str(seed)]["visible_revision_B_cues"]),
                ("corrupt_revision_B_cue", audit["independent_oracle"]["per_seed"][str(seed)]["corrupt_revision_B_cue"])
            )} and stage["arms"]["correction_packet"]["accuracy"]
            == audit["independent_oracle"]["per_seed"][str(seed)]["conditions"][condition]["global_accuracy"]
            and stage["arms"]["correction_packet"]["changed_signs"]
            == audit["independent_oracle"]["per_seed"][str(seed)]["conditions"][condition]["changed_signs"]
            for seed in seeds for condition, stage in per_seed[str(seed)]["conditions"].items()),
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
            run_arm(*_adapt([row for row in rows if row["seed"] == seed], protocol),
                    *local_maps(protocol, seed, _adapt([row for row in rows if row["seed"] == seed], protocol)[0]["B_map"], "full")[:2],
                    "correction_packet")["prediction_trace_sha256"]
            == per_seed[str(seed)]["conditions"]["full"]["arms"]["correction_packet"]["prediction_trace_sha256"]
            for seed in seeds),
    }
    report = {
        "schema": "sara-local-credit-packet-mixed-error-revision-heldout-result-v1",
        "protocol_sha256": execution["protocol_sha256"],
        "execution_sha256": EXECUTION_SHA256,
        "rows_sha256": execution["rows_sha256"],
        "accuracy_by_condition": summary,
        "full_control_accuracy": controls,
        "heldout_checks": checks,
        "harness_checks": harness,
        "heldout_gate_passed": all(checks.values()),
        "harness_passed": all(harness.values()),
        "per_seed": per_seed,
        "heldout_consumed": True,
        "production_authorized": False,
    }
    Path(ensure_parent_directory(OUTPUT)).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accuracy_by_condition": summary, "full_control_accuracy": controls,
                      "heldout_checks": checks, "harness_checks": harness}, sort_keys=True))
    return 0 if report["heldout_gate_passed"] and report["harness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
