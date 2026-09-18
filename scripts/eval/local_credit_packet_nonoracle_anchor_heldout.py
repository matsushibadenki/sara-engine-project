#!/usr/bin/env python3
"""One-shot held-out scorer for label-blind Local Credit Packet acquisition."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_allowed_output_path, processed_data_path, workspace_path,
)

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_nonoracle_anchor_acquisition_heldout_v1.json"
))
EXECUTION = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_nonoracle_anchor_acquisition_heldout_execution_v1.json"
))
ROWS = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_nonoracle_anchor_acquisition_heldout_rows_v1.jsonl"
))
AUDIT = Path(workspace_path(
    "evaluation", "local_credit_packet_nonoracle_anchor_acquisition_heldout_audit_v1.json"
))
OUTPUT = Path(workspace_path(
    "evaluation", "local_credit_packet_nonoracle_anchor_acquisition_heldout_v1.json"
))
CANDIDATE = ROOT / "scripts/eval/local_credit_packet_nonoracle_anchor_acquisition.py"
CORE = ROOT / "src/sara_engine/evaluation/local_credit_packet_multihop.py"
MATERIALIZER = ROOT / "scripts/eval/local_credit_packet_nonoracle_anchor_heldout_materialize.py"
PARENT_PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_nonoracle_anchor_acquisition_v1.json"
))
PARENT_RESULT = Path(workspace_path(
    "evaluation", "local_credit_packet_nonoracle_anchor_acquisition_development_v1.json"
))
PROTOCOL_SHA256 = "6afee9eb95988e03a38af7af1870abc0bb7bf3888c86d75fa9a560a636d0afa1"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> tuple[dict, dict, list[dict], dict]:
    if sys.version_info[:2] != (3, 10):
        raise ValueError("Held-out scoring requires CPython 3.10")
    if OUTPUT.exists() or Path(str(OUTPUT) + ".lock").exists():
        raise ValueError("Held-out scoring is already reserved or completed")
    if _sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Held-out protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    execution = json.loads(EXECUTION.read_text())
    if (protocol.get("schema") != "sara-local-credit-packet-nonoracle-anchor-acquisition-heldout-preregistration-v1"
            or execution.get("schema") != "sara-local-credit-packet-nonoracle-anchor-acquisition-heldout-execution-v1"
            or execution.get("protocol_sha256") != PROTOCOL_SHA256
            or execution.get("pre_execution_audit_passed") is not True
            or execution.get("candidate_scored") is not False):
        raise ValueError("Held-out execution boundary changed")
    pinned = (
        (CANDIDATE, "candidate_source_sha256"),
        (CORE, "candidate_core_sha256"),
        (MATERIALIZER, "independent_materializer_sha256"),
        (ROWS, "heldout_rows_sha256"),
        (AUDIT, "pre_execution_audit_sha256"),
    )
    if any(_sha(path) != execution[key] for path, key in pinned):
        raise ValueError("Pinned held-out source or data changed")
    if (_sha(PARENT_PROTOCOL) != protocol["parent_development_protocol_sha256"]
            or _sha(PARENT_RESULT) != protocol["parent_development_result_sha256"]
            or execution["candidate_source_sha256"] != protocol["candidate_source_sha256"]
            or _sha(Path(__file__)) != execution["scorer_source_sha256"]):
        raise ValueError("Candidate, parent, or scorer identity changed")
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    audit = json.loads(AUDIT.read_text())
    if (audit.get("passed") is not True or audit.get("candidate_scored") is not False
            or audit.get("rows_sha256") != execution["heldout_rows_sha256"]
            or audit.get("protocol_sha256") != PROTOCOL_SHA256):
        raise ValueError("Pre-execution audit was not clean")
    materializer = runpy.run_path(str(MATERIALIZER))
    replay = materializer["independent_audit"](rows, protocol)
    if not replay["passed"] or replay["changed_sign_strata"] != audit["changed_sign_strata"]:
        raise ValueError("Independent audit does not replay")
    return protocol, execution, rows, materializer


def _seed_data(rows: list[dict], protocol: dict, seed: int) -> tuple[dict, list[tuple[int, dict]], set[int]]:
    identity = protocol["identity"]
    training = [row for row in rows if row["seed"] == seed and row["phase"] == "training"]
    test = [row for row in rows if row["seed"] == seed and row["phase"] == "test"]
    A_map = {row["A_cue"]: row["A_target"] for row in test}
    B_map = {row["B_cue"]: row["B_local_target"] for row in test}
    ranks = {cue: sha256(f'{identity["namespace"]}|initial_wrong|{seed}|{cue}'.encode()).digest()
             for cue in identity["B_cues"]}
    wrong = set(sorted(ranks, key=ranks.__getitem__)[:4])
    initial = {cue: B_map[cue] ^ int(cue in wrong) for cue in B_map}
    selected = []
    seen = Counter()
    for position, row in enumerate(training, 1):
        if seen[row["A_cue"]] < 2:
            selected.append((position, row))
            seen[row["A_cue"]] += 1
    data = {
        "A_map": A_map, "B_map": B_map, "initial": initial,
        "events": [(row["original_index"], row["A_cue"], row["B_cue"],
                    row["A_action"], row["B_action"], row["success"])
                   for row in training],
    }
    return data, selected, wrong


def _oracle_digests(oracle: dict, selected: list[tuple[int, dict]],
                    wrong: set[int], B_map: dict[int, int], arm: str,
                    events: list[tuple[int, int, int, int, int, int]]) -> dict:
    touched = {(row["A_cue"], row["A_action"]) for _, row in selected}
    if arm != "prior_only":
        for _, row in selected:
            if row["B_cue"] in wrong or arm == "unconditional_two_step":
                touched.add((row["A_cue"], row["A_action"]
                             ^ int(arm == "route_shuffle_two_step")))
    weights = sorted((cue, branch, oracle["per_cue"][str(cue)]["weights"][branch])
                     for cue, branch in touched)
    predictions = [(cue, B_cue, row["prediction"],
                    row["prediction"] ^ B_map[B_cue])
                   for cue, row in sorted((int(cue), row)
                                          for cue, row in oracle["per_cue"].items())
                   for B_cue in sorted(B_map)]
    return {
        "weights": sha256(json.dumps(weights).encode()).hexdigest(),
        "predictions": sha256(json.dumps(predictions).encode()).hexdigest(),
        "forward": sha256(json.dumps(events).encode()).hexdigest(),
    }


def evaluate(protocol: dict, rows: list[dict], materializer: dict) -> dict:
    candidate = runpy.run_path(str(CANDIDATE))
    per_seed = {}
    oracle_checks = {}
    resource_checks = {}
    for seed in protocol["identity"]["seeds"]:
        data, selected, wrong = _seed_data(rows, protocol, seed)
        seed_arms = {}
        for arm in protocol["evaluation"]["arms"]:
            result = candidate["run_arm"](data, arm)
            oracle = materializer["_oracle_arm"](
                selected, data["A_map"], data["B_map"], wrong, arm)
            digests = _oracle_digests(oracle, selected, wrong, data["B_map"],
                                      arm, data["events"])
            oracle_checks[f"{seed}:{arm}"] = (
                result["replay_packets"] == oracle["packets"]
                and result["weight_trace_sha256"] == digests["weights"]
                and result["prediction_trace_sha256"] == digests["predictions"]
                and result["forward_trace_sha256"] == digests["forward"]
                and all(result["per_cue"][cue]["target"] == row["target"]
                        and result["per_cue"][cue]["correct"] == row["correct"]
                        and result["per_cue"][cue]["changed_sign_count"] == row["changed"]
                        and result["per_cue"][cue]["tie"] == (row["weights"][0] == row["weights"][1])
                        for cue, row in oracle["per_cue"].items()))
            budget = protocol["resource_budgets"]
            resource_checks[f"{seed}:{arm}"] = (
                result["maximum_A_anchors"] <= budget["maximum_A_anchors"]
                and result["maximum_B_anchors"] <= budget["maximum_B_anchors"]
                and result["maximum_A_eligibility_entries"] <= budget["maximum_A_eligibility_entries"]
                and result["A_features"] <= budget["maximum_A_features"]
                and result["B_features"] <= budget["maximum_B_features"]
                and result["maximum_packet_bytes"] <= budget["maximum_packet_bytes"]
                and result["replay_lookups"] <= budget["maximum_replay_lookups"]
                and result["replay_packets"] <= budget["maximum_replay_packets"]
                and result["A_updates"] - result["early_packets"]
                <= budget["maximum_local_steps_per_packet"] * result["replay_packets"]
                and result["peak_state_bytes"] <= budget["maximum_total_state_bytes"])
            seed_arms[arm] = result
        per_seed[str(seed)] = seed_arms
    return {"per_seed": per_seed, "oracle_checks": oracle_checks,
            "resource_checks": resource_checks}


def decision(protocol: dict, scored: dict) -> dict:
    per_seed = scored["per_seed"]
    arms = protocol["evaluation"]["arms"]
    mean = {arm: sum(seed[arm]["accuracy"] for seed in per_seed.values()) / len(per_seed)
            for arm in arms}
    strata = {}
    for arm in arms:
        strata[arm] = {}
        for changed in range(3):
            strata[arm][str(changed)] = {}
            for target in (0, 1):
                selected = [row for seed in per_seed.values()
                            for row in seed[arm]["per_cue"].values()
                            if row["changed_sign_count"] == changed and row["target"] == target]
                strata[arm][str(changed)][str(target)] = {
                    "count": len(selected),
                    "accuracy": (sum(row["correct"] for row in selected) / len(selected)
                                 if selected else None),
                }
    selective_packets = sum(seed["selective_two_step"]["replay_packets"]
                            for seed in per_seed.values())
    unconditional_packets = sum(seed["unconditional_two_step"]["replay_packets"]
                                for seed in per_seed.values())
    reduction = 1 - selective_packets / unconditional_packets
    a = protocol["acceptance"]
    checks = {
        "selective_mean": mean["selective_two_step"] >= a["selective_mean_accuracy_minimum"],
        "unconditional_mean": mean["unconditional_two_step"] >= a["unconditional_mean_accuracy_minimum"],
        "selective_each_seed": all(seed["selective_two_step"]["accuracy"]
                                   >= a["minimum_each_seed_selective_accuracy"]
                                   for seed in per_seed.values()),
        "two_change_support": all(strata["selective_two_step"]["2"][str(target)]["count"]
                                  >= a["minimum_two_change_cues_per_target_pooled"]
                                  for target in (0, 1)),
        "two_change_selective_by_target": all(
            strata["selective_two_step"]["2"][str(target)]["accuracy"]
            >= a["two_change_selective_accuracy_each_target_minimum"]
            for target in (0, 1)),
        "simple_tie_zero_target_one": (
            strata["simple_tie_zero"]["2"]["1"]["accuracy"]
            <= a["two_change_simple_tie_zero_target_one_accuracy_maximum"]),
        "simple_tie_one_target_zero": (
            strata["simple_tie_one"]["2"]["0"]["accuracy"]
            <= a["two_change_simple_tie_one_target_zero_accuracy_maximum"]),
        "packet_reduction": reduction >= a["minimum_replay_packet_reduction_fraction"],
        "prediction_identity": all(
            seed["selective_two_step"]["prediction_trace_sha256"]
            == seed["unconditional_two_step"]["prediction_trace_sha256"]
            for seed in per_seed.values()),
        "wrong_sign_drop": mean["selective_two_step"] - mean["sign_shuffle_two_step"]
        >= a["minimum_wrong_sign_or_route_accuracy_drop"],
        "wrong_route_drop": mean["selective_two_step"] - mean["route_shuffle_two_step"]
        >= a["minimum_wrong_sign_or_route_accuracy_drop"],
        "oracle_replay": all(scored["oracle_checks"].values()),
        "resources": all(scored["resource_checks"].values()),
    }
    return {"mean_accuracy": mean, "strata": strata,
            "selective_replay_packets": selective_packets,
            "unconditional_replay_packets": unconditional_packets,
            "replay_packet_reduction_fraction": reduction,
            "checks": checks, "heldout_gate_passed": all(checks.values())}


def main() -> int:
    protocol, execution, rows, materializer = verify_inputs()
    output = Path(ensure_allowed_output_path(str(OUTPUT)))
    output.parent.mkdir(parents=True, exist_ok=True)
    execution_sha256 = _sha(EXECUTION)
    with Path(str(output) + ".lock").open("xb") as stream:
        stream.write((execution_sha256 + "\n").encode("ascii"))
    scored = evaluate(protocol, rows, materializer)
    verdict = decision(protocol, scored)
    report = {
        "schema": "sara-local-credit-packet-nonoracle-anchor-acquisition-heldout-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "execution_sha256": execution_sha256,
        "rows_sha256": execution["heldout_rows_sha256"],
        "audit_sha256": execution["pre_execution_audit_sha256"],
        **verdict,
        **scored,
        "test_updates": False,
        "production_authorized": False,
        "real_event_credit_claim_authorized": False,
    }
    payload = (json.dumps(report, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({key: report[key] for key in
                      ("mean_accuracy", "selective_replay_packets",
                       "unconditional_replay_packets", "replay_packet_reduction_fraction",
                       "checks", "heldout_gate_passed")}, sort_keys=True))
    return 0 if report["heldout_gate_passed"] else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--preflight"]:
        verify_inputs()
        print("Held-out identity preflight passed; no candidate scored")
    elif sys.argv[1:]:
        raise SystemExit("Unknown argument")
    else:
        raise SystemExit(main())
