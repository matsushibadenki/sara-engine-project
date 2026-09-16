#!/usr/bin/env python3
"""Audit three-stage overlapping credit before any replay learner exists."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

PARENT = Path(processed_data_path("benchmark_fixtures", "local_credit_packet_three_stage_v1.json"))
SUPPLEMENT = Path(processed_data_path("benchmark_fixtures", "local_credit_packet_three_stage_materialization_v1.json"))
PARENT_SHA256 = "ea61d37c8cf55d8c125d9b1be7c359fd659b9002f50fe775b449cf7cf9daf709"
SUPPLEMENT_SHA256 = "afbe5cb3eb5c2703bab3fb7ea17dce8b0f0bec9ef0934c0f8dd4fbc3767218df"


def _digest(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def _target_rule(seed: int) -> dict[int, int]:
    contexts = list(range(8))
    random.Random(seed + 31).shuffle(contexts)
    return {context: int(position >= 4) for position, context in enumerate(contexts)}


def generate_rows(parent: dict, supplement: dict) -> list[dict]:
    identity = parent["identity"]
    schedule = supplement["schedule"]
    contexts = schedule["contexts"]
    wave_size = schedule["wave_size"]
    rows = []
    for seed in identity["seeds"]:
        targets = _target_rule(seed)
        for split, count in (
            ("training", identity["training_episode_count_per_seed"]),
            ("development", identity["development_episode_count_per_seed"]),
        ):
            if count % wave_size:
                raise ValueError("Split count must contain complete waves")
            for index in range(count):
                wave, slot = divmod(index, wave_size)
                context = contexts[slot]
                action = wave % 2
                alias = ("route-alpha", "route-beta")[(action + (wave // 2) % 2) % 2]
                delay = schedule["delay_cycle_steps"][wave % 3]
                start = wave * schedule["wave_spacing_steps"]
                source = _digest(identity["namespace"], split, seed, index, "source")
                outcome = _digest(identity["namespace"], split, seed, index, "outcome")
                route = _digest(identity["namespace"], "route", action, alias)
                target = targets[context]
                rows.append({
                    "identity": f"{identity['namespace']}:{split}:{seed}:{index}",
                    "split": split,
                    "seed": seed,
                    "wave": wave,
                    "context_id": context,
                    "forced_action": action,
                    "route_alias": alias,
                    "route_digest": route,
                    "source_event": source,
                    "outcome_id": outcome,
                    "source_step": start,
                    "intermediate_step": start + schedule["intermediate_offset_steps"],
                    "anchor_step": start + schedule["anchor_offset_steps"],
                    "direct_expiry_step": start + parent["task"]["direct_eligibility_ttl"],
                    "outcome_step": start + delay,
                    "anchor_expiry_step": start + delay + 1,
                    "target_branch": target,
                    "outcome_success": int(action == target),
                })
    return rows


def majority_accuracy(rows: list[dict], key) -> float:
    groups: dict[object, Counter] = defaultdict(Counter)
    for row in rows:
        groups[key(row)][row["target_branch"]] += 1
    return sum(max(counts.values()) for counts in groups.values()) / len(rows)


def audit_rows(rows: list[dict], parent: dict, supplement: dict) -> dict:
    task = parent["task"]
    limits = supplement["pre_candidate_audit"]
    by_split_seed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    by_context: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    by_wave: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_split_seed[(row["split"], row["seed"])].append(row)
        by_context[(row["split"], row["seed"], row["context_id"])].append(row)
        by_wave[(row["split"], row["seed"], row["wave"])].append(row)
    source_ids = [row["source_event"] for row in rows]
    outcome_ids = [row["outcome_id"] for row in rows]
    target_by_seed_context: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in rows:
        target_by_seed_context[(row["seed"], row["context_id"])].add(row["target_branch"])
    occupancy_peaks = []
    direct_available_at_outcome = []
    anchor_available_at_outcome = []
    for wave_rows in by_wave.values():
        starts = {row["source_step"] for row in wave_rows}
        outcomes = {row["outcome_step"] for row in wave_rows}
        occupancy_peaks.append(len(wave_rows))
        if len(starts) != 1 or len(outcomes) != 1:
            raise ValueError("Wave events do not overlap")
        for row in wave_rows:
            direct_available_at_outcome.append(row["outcome_step"] <= row["direct_expiry_step"])
            anchor_available_at_outcome.append(
                row["anchor_step"] <= row["outcome_step"] < row["anchor_expiry_step"]
            )
    final_accuracy = majority_accuracy(
        rows,
        lambda row: (
            row["outcome_step"] - row["source_step"],
            row["route_alias"],
        ),
    )
    route_accuracy = majority_accuracy(rows, lambda row: row["route_digest"])
    source_bit_accuracy = majority_accuracy(rows, lambda row: int(row["source_event"][-1], 16) & 1)
    outcome_bit_accuracy = majority_accuracy(rows, lambda row: int(row["outcome_id"][-1], 16) & 1)
    checks = {
        "row_count": len(rows) == len(parent["identity"]["seeds"]) * (
            parent["identity"]["training_episode_count_per_seed"]
            + parent["identity"]["development_episode_count_per_seed"]
        ),
        "target_balance": all(
            Counter(row["target_branch"] for row in group) == Counter({0: len(group) // 2, 1: len(group) // 2})
            for group in by_split_seed.values()
        ),
        "forced_action_balance_per_context": all(
            Counter(row["forced_action"] for row in group)
            == Counter({0: len(group) // 2, 1: len(group) // 2})
            for group in by_context.values()
        ),
        "stable_target_across_splits": all(len(values) == 1 for values in target_by_seed_context.values()),
        "complete_waves": all(len(group) == 8 for group in by_wave.values()),
        "actual_overlap": min(occupancy_peaks) >= limits["minimum_concurrent_eligibilities"]
        and max(occupancy_peaks) <= limits["maximum_concurrent_eligibilities"],
        "anchor_occupancy_bounded": max(occupancy_peaks) <= limits["maximum_anchor_occupancy"],
        "all_direct_traces_expired": not any(direct_available_at_outcome),
        "all_anchors_available": all(anchor_available_at_outcome),
        "next_wave_after_outcomes": all(
            row["outcome_step"] < (row["wave"] + 1) * supplement["schedule"]["wave_spacing_steps"]
            for row in rows
        ),
        "stage_order": all(
            row["source_step"] < row["intermediate_step"] < row["anchor_step"]
            < row["direct_expiry_step"] < row["outcome_step"] < row["anchor_expiry_step"]
            for row in rows
        ),
        "outcome_rule": all(row["outcome_success"] == int(row["forced_action"] == row["target_branch"]) for row in rows),
        "unique_source_and_outcome_ids": len(set(source_ids)) == len(rows)
        and len(set(outcome_ids)) == len(rows),
        "split_identity_disjoint": len({row["identity"] for row in rows}) == len(rows),
        "route_digest_not_predictive": route_accuracy <= limits["route_digest_majority_accuracy_maximum"],
        "final_stage_insufficient": final_accuracy <= limits["final_stage_majority_accuracy_maximum"],
        "source_bit_not_predictive": source_bit_accuracy <= limits["opaque_id_low_bit_accuracy_maximum"],
        "outcome_bit_not_predictive": outcome_bit_accuracy <= limits["opaque_id_low_bit_accuracy_maximum"],
    }
    return {
        "checks": checks,
        "diagnostics": {
            "wave_count": len(by_wave),
            "peak_concurrent_eligibilities": max(occupancy_peaks),
            "minimum_concurrent_eligibilities": min(occupancy_peaks),
            "peak_anchor_entries": max(occupancy_peaks),
            "direct_available_at_outcome_count": sum(direct_available_at_outcome),
            "anchor_available_at_outcome_count": sum(anchor_available_at_outcome),
            "final_stage_majority_accuracy": final_accuracy,
            "route_digest_majority_accuracy": route_accuracy,
            "source_id_low_bit_majority_accuracy": source_bit_accuracy,
            "outcome_id_low_bit_majority_accuracy": outcome_bit_accuracy,
        },
    }


def main() -> int:
    parent_raw = PARENT.read_bytes()
    supplement_raw = SUPPLEMENT.read_bytes()
    if hashlib.sha256(parent_raw).hexdigest() != PARENT_SHA256:
        raise ValueError("Frozen parent protocol changed")
    if hashlib.sha256(supplement_raw).hexdigest() != SUPPLEMENT_SHA256:
        raise ValueError("Frozen materialization protocol changed")
    parent = json.loads(parent_raw)
    supplement = json.loads(supplement_raw)
    if supplement["parent_protocol_sha256"] != PARENT_SHA256:
        raise ValueError("Materialization parent mismatch")
    rows = generate_rows(parent, supplement)
    audit = audit_rows(rows, parent, supplement)
    row_output = Path(processed_data_path("benchmark_fixtures", "local_credit_packet_three_stage_rows_v1.jsonl"))
    row_output = Path(ensure_parent_directory(row_output))
    row_output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = {
        "schema": "sara-local-credit-packet-three-stage-identifiability-v1",
        "parent_protocol_sha256": PARENT_SHA256,
        "materialization_protocol_sha256": SUPPLEMENT_SHA256,
        "rows_sha256": hashlib.sha256(row_output.read_bytes()).hexdigest(),
        "row_count": len(rows),
        **audit,
        "passed": all(audit["checks"].values()),
        "candidate_execution_authorized": all(audit["checks"].values()),
        "candidate_implemented": False,
        "heldout_consumed": False,
        "production_authorized": False,
    }
    output = Path(ensure_parent_directory(workspace_path("evaluation", "local_credit_packet_three_stage_identifiability.json")))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), **report}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
