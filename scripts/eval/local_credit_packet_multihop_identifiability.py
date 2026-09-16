#!/usr/bin/env python3
"""Audit multi-hop task information flow before candidate implementation."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
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

PARENT = Path(processed_data_path("benchmark_fixtures", "local_credit_packet_multihop_v1.json"))
SUPPLEMENT = Path(processed_data_path("benchmark_fixtures", "local_credit_packet_multihop_materialization_v1.json"))
PARENT_SHA256 = "be5a941959c7919efb236b16495e604d8822716fbeeeef8fcab315a42aa38fb2"
SUPPLEMENT_SHA256 = "ef3cf6f26122ca1964c7bf49a53ef3bc0d7c4430b8ec7689dd05cddca86f1bc5"


def _opaque_id(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def _balanced_map(seed: int, offset: int) -> dict[int, int]:
    cues = list(range(8))
    random.Random(seed + offset).shuffle(cues)
    return {cue: int(index >= 4) for index, cue in enumerate(cues)}


def generate_rows(parent: dict) -> list[dict]:
    identity = parent["identity"]
    namespace = identity["namespace"]
    rows = []
    for seed in identity["seeds"]:
        A_map = _balanced_map(seed, 101)
        B_map = _balanced_map(seed, 202)
        calibration = []
        for index in range(parent["training_schedule"]["B_local_calibration_rows_per_seed"]):
            B_cue = index % 8
            calibration.append({
                "identity": f"{namespace}:B_calibration:{seed}:{index}",
                "phase": "B_calibration",
                "seed": seed,
                "B_cue": B_cue,
                "B_local_target": B_map[B_cue],
                "source_event": _opaque_id(namespace, "B_calibration", seed, index, "source"),
            })
        random.Random(seed + 303).shuffle(calibration)
        rows.extend(calibration)
        main = []
        factorial = itertools.product(range(8), range(8), (0, 1), (0, 1), range(2))
        for index, (A_cue, B_cue, A_action, B_action, repeat) in enumerate(factorial):
            A_target = A_map[A_cue]
            B_target = B_map[B_cue]
            global_target = A_target ^ B_target
            main.append({
                "identity": f"{namespace}:main_training:{seed}:{index}",
                "phase": "main_training",
                "seed": seed,
                "A_cue": A_cue,
                "B_cue": B_cue,
                "forced_A_action": A_action,
                "forced_B_action": B_action,
                "repeat": repeat,
                "A_target": A_target,
                "B_local_target": B_target,
                "global_target": global_target,
                "global_success": int(B_action == global_target),
                "source_event": _opaque_id(namespace, "main_training", seed, index, "source"),
                "outcome_id": _opaque_id(namespace, "main_training", seed, index, "outcome"),
            })
        random.Random(seed + 404).shuffle(main)
        rows.extend(main)
        development = []
        for index, (A_cue, B_cue, repeat) in enumerate(
            itertools.product(range(8), range(8), range(2))
        ):
            A_target = A_map[A_cue]
            B_target = B_map[B_cue]
            development.append({
                "identity": f"{namespace}:development:{seed}:{index}",
                "phase": "development",
                "seed": seed,
                "A_cue": A_cue,
                "B_cue": B_cue,
                "repeat": repeat,
                "A_target": A_target,
                "B_local_target": B_target,
                "global_target": A_target ^ B_target,
                "source_event": _opaque_id(namespace, "development", seed, index, "source"),
                "outcome_id": _opaque_id(namespace, "development", seed, index, "outcome"),
            })
        random.Random(seed + 505).shuffle(development)
        rows.extend(development)
    return rows


def majority_accuracy(rows: list[dict], feature, label) -> float:
    groups: dict[object, Counter] = defaultdict(Counter)
    for row in rows:
        groups[feature(row)][label(row)] += 1
    return sum(max(counts.values()) for counts in groups.values()) / len(rows)


def audit_rows(rows: list[dict], parent: dict, supplement: dict) -> dict:
    seeds = parent["identity"]["seeds"]
    limits = supplement["pre_candidate_audit"]
    by_seed_phase: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_seed_phase[(row["seed"], row["phase"])].append(row)
    main = [row for row in rows if row["phase"] == "main_training"]
    development = [row for row in rows if row["phase"] == "development"]
    direct_shortcut_accuracy = majority_accuracy(
        main,
        lambda row: (row["forced_A_action"], row["forced_B_action"], row["global_success"]),
        lambda row: row["A_target"],
    )
    outcome_only_accuracy = majority_accuracy(
        main, lambda row: row["global_success"], lambda row: row["A_target"]
    )
    source_bit_accuracy = majority_accuracy(
        main, lambda row: int(row["source_event"][-1], 16) & 1,
        lambda row: row["A_target"],
    )
    outcome_bit_accuracy = majority_accuracy(
        main, lambda row: int(row["outcome_id"][-1], 16) & 1,
        lambda row: row["A_target"],
    )
    decoded_A = [
        (row["forced_B_action"] ^ (1 - row["global_success"]))
        ^ row["B_local_target"]
        for row in main
    ]
    A_map_stability = all(
        len({row["A_target"] for row in main + development
             if row["seed"] == seed and row["A_cue"] == cue}) == 1
        for seed in seeds for cue in range(8)
    )
    B_map_stability = all(
        len({row["B_local_target"] for row in rows
             if row["seed"] == seed and row["B_cue"] == cue}) == 1
        for seed in seeds for cue in range(8)
    )
    action_completeness = all(
        Counter((row["forced_A_action"], row["forced_B_action"])
                for row in by_seed_phase[(seed, "main_training")]
                if row["A_cue"] == A_cue and row["B_cue"] == B_cue)
        == Counter({(0, 0): 2, (0, 1): 2, (1, 0): 2, (1, 1): 2})
        for seed in seeds for A_cue in range(8) for B_cue in range(8)
    )
    all_ids = [row["identity"] for row in rows]
    source_ids = [row["source_event"] for row in rows]
    outcome_ids = [row["outcome_id"] for row in rows if "outcome_id" in row]
    checks = {
        "row_count": len(rows) == len(seeds) * (128 + 512 + 128),
        "balanced_private_maps": all(
            Counter(_balanced_map(seed, offset).values()) == Counter({0: 4, 1: 4})
            for seed in seeds for offset in (101, 202)
        ),
        "stable_private_maps": A_map_stability and B_map_stability,
        "complete_forced_action_factorial": action_completeness,
        "calibration_count": all(len(by_seed_phase[(seed, "B_calibration")]) == 128 for seed in seeds),
        "train_target_and_success_balance": all(
            Counter(row["global_target"] for row in by_seed_phase[(seed, "main_training")])
            == Counter({0: 256, 1: 256})
            and Counter(row["global_success"] for row in by_seed_phase[(seed, "main_training")])
            == Counter({0: 256, 1: 256})
            for seed in seeds
        ),
        "development_target_balance": all(
            Counter(row["global_target"] for row in by_seed_phase[(seed, "development")])
            == Counter({0: 64, 1: 64})
            for seed in seeds
        ),
        "direct_shortcut_uninformative": direct_shortcut_accuracy <= limits["shortcut_majority_accuracy_maximum"],
        "outcome_only_uninformative": outcome_only_accuracy <= limits["outcome_only_majority_accuracy_maximum"],
        "B_private_target_enables_exact_A_decoding": all(
            decoded == row["A_target"] for decoded, row in zip(decoded_A, main)
        ),
        "source_id_probe": source_bit_accuracy <= limits["opaque_id_low_bit_accuracy_maximum"],
        "outcome_id_probe": outcome_bit_accuracy <= limits["opaque_id_low_bit_accuracy_maximum"],
        "identities_disjoint": len(all_ids) == len(set(all_ids)),
        "source_and_outcome_ids_unique": len(source_ids) == len(set(source_ids))
        and len(outcome_ids) == len(set(outcome_ids)),
        "candidate_order_declared": limits["no_candidate_implementation_before_audit"],
        "heldout_closed": parent["identity"]["heldout_materialized"] is False
        and parent["identity"]["heldout_consumed"] is False,
    }
    return {
        "checks": checks,
        "diagnostics": {
            "direct_shortcut_A_target_majority_accuracy": direct_shortcut_accuracy,
            "outcome_only_A_target_majority_accuracy": outcome_only_accuracy,
            "source_id_low_bit_A_target_accuracy": source_bit_accuracy,
            "outcome_id_low_bit_A_target_accuracy": outcome_bit_accuracy,
            "B_private_target_decoding_accuracy": sum(
                decoded == row["A_target"] for decoded, row in zip(decoded_A, main)
            ) / len(main),
        },
    }


def main() -> int:
    parent_raw = PARENT.read_bytes()
    supplement_raw = SUPPLEMENT.read_bytes()
    if hashlib.sha256(parent_raw).hexdigest() != PARENT_SHA256:
        raise ValueError("Frozen multi-hop protocol changed")
    if hashlib.sha256(supplement_raw).hexdigest() != SUPPLEMENT_SHA256:
        raise ValueError("Frozen materialization supplement changed")
    parent = json.loads(parent_raw)
    supplement = json.loads(supplement_raw)
    if supplement["parent_protocol_sha256"] != PARENT_SHA256:
        raise ValueError("Materialization parent mismatch")
    rows = generate_rows(parent)
    audit = audit_rows(rows, parent, supplement)
    output_rows = Path(ensure_parent_directory(processed_data_path(
        "benchmark_fixtures", "local_credit_packet_multihop_rows_v1.jsonl"
    )))
    output_rows.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = {
        "schema": "sara-local-credit-packet-multihop-identifiability-v1",
        "parent_protocol_sha256": PARENT_SHA256,
        "materialization_sha256": SUPPLEMENT_SHA256,
        "rows_sha256": hashlib.sha256(output_rows.read_bytes()).hexdigest(),
        "row_count": len(rows),
        **audit,
        "passed": all(audit["checks"].values()),
        "candidate_execution_authorized": all(audit["checks"].values()),
        "candidate_implemented_now": (
            ROOT / "src" / "sara_engine" / "evaluation" / "local_credit_packet_multihop.py"
        ).exists(),
        "heldout_consumed": False,
        "production_authorized": False,
    }
    output_report = Path(ensure_parent_directory(workspace_path(
        "evaluation", "local_credit_packet_multihop_identifiability.json"
    )))
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_report), **report}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
