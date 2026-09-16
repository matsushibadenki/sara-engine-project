#!/usr/bin/env python3
"""Run the corrected, fresh-data noisy/late B-evidence development protocol."""
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
from scripts.eval.local_credit_packet_noisy_late_B_evidence import (  # noqa: E402
    _rank, digest, materialize, run_arm, schedule,
)

V1_PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_noisy_late_B_evidence_v1.json"
))
V2_PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_noisy_late_B_evidence_v2.json"
))
V2_SHA256 = "bfacd6b9f680ca7885698026a72d019e957ff58fd82893e7b459993334e200fb"
V1_RUNNER = ROOT / "scripts/eval/local_credit_packet_noisy_late_B_evidence.py"
V1_RESULT = Path(workspace_path(
    "evaluation", "local_credit_packet_noisy_late_B_evidence_development_v1.json"
))
OUTPUT = Path(workspace_path(
    "evaluation", "local_credit_packet_noisy_late_B_evidence_development_v2.json"
))


def audit_selection(protocol: dict, seed: int, rows: dict,
                    training: list[tuple], flips: dict[str, set[int]]) -> None:
    namespace = protocol["identity"]["namespace"]
    expected = [row for _, row in sorted(
        enumerate(rows["training"]),
        key=lambda pair: _rank(namespace + "|order", seed, pair[0]),
    )]
    if training != expected:
        raise ValueError("Training order disagrees with frozen selector")
    for cue in protocol["identity"]["B_cues"]:
        indices = sorted(
            (index for index, row in enumerate(training) if row[1] == cue),
            key=lambda index: _rank(namespace + "|flip", seed, cue, index),
        )
        if len(indices) != 64:
            raise ValueError("Unbalanced cue in frozen selector")
        for name, count in (("clean_immediate", 0),
                            ("quarter_noise_immediate", 16),
                            ("half_noise_immediate", 32)):
            if set(indices) & flips[name] != set(indices[:count]):
                raise ValueError("Flip schedule disagrees with frozen selector")


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("V2 development result already exists; refusing to rerun")
    if hashlib.sha256(V2_PROTOCOL.read_bytes()).hexdigest() != V2_SHA256:
        raise ValueError("Frozen V2 protocol changed")
    supplement = json.loads(V2_PROTOCOL.read_text())
    for path, key in (
        (V1_PROTOCOL, "parent_v1_protocol_sha256"),
        (V1_RUNNER, "parent_v1_runner_sha256"),
        (V1_RESULT, "parent_v1_result_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != supplement[key]:
            raise ValueError(f"Frozen V1 parent changed: {path.name}")
    protocol = json.loads(V1_PROTOCOL.read_text())
    protocol["identity"] = supplement["identity"]
    seeds = protocol["identity"]["seeds"]
    per_seed = {}
    for seed in seeds:
        rows = materialize(protocol, seed)
        training, flips = schedule(protocol, seed, rows)
        audit_selection(protocol, seed, rows, training, flips)
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
        "selector_audited": True,
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
            _replay(protocol, seed) == per_seed[str(seed)]["conditions"][
                "quarter_noise_immediate"]["arms"]["local_credit_packet"]["prediction_trace_sha256"]
            for seed in seeds),
    }
    report = {"schema": "sara-local-credit-packet-noisy-late-B-evidence-development-v2",
              "protocol_sha256": V2_SHA256,
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


def _replay(protocol: dict, seed: int) -> str:
    rows = materialize(protocol, seed)
    training, flips = schedule(protocol, seed, rows)
    return run_arm(rows, training, flips["quarter_noise_immediate"], 0,
                   "local_credit_packet")["prediction_trace_sha256"]


if __name__ == "__main__":
    raise SystemExit(main())
