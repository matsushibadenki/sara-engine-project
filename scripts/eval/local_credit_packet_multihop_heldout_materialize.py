#!/usr/bin/env python3
"""Independently generate and audit the frozen multi-hop held-out task."""
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

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_multihop_heldout_v1.json"
))
PROTOCOL_SHA256 = "5b542db726cf652c69158fa15cb007c3afa06ff615016a201bde9c809470402f"
ROWS = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_multihop_heldout_rows_v1.jsonl"
))


def _digest(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _generate_map(seed: int, cues: list[int], tag: str) -> dict[int, int]:
    ranked = sorted(cues, key=lambda cue: hashlib.sha256(
        f"{seed}|{cue}|{tag}".encode()
    ).digest())
    return {cue: int(position >= len(cues) // 2) for position, cue in enumerate(ranked)}


def _oracle_map(seed: int, cues: list[int], tag: str) -> dict[int, int]:
    # Independently reconstruct the rank order without calling the generator.
    ranks = {
        cue: int.from_bytes(
            hashlib.new("sha256", (str(seed) + "|" + str(cue) + "|" + tag).encode()).digest(),
            "big",
        )
        for cue in cues
    }
    order = sorted(ranks, key=ranks.__getitem__)
    midpoint = len(order) // 2
    return {cue: int(order.index(cue) >= midpoint) for cue in cues}


def generate(protocol: dict) -> list[dict]:
    identity = protocol["identity"]
    namespace = identity["namespace"]
    rows = []
    for seed in identity["seeds"]:
        A_map = _generate_map(seed, identity["A_cues"], "A-map")
        B_map = _generate_map(seed, identity["B_cues"], "B-map")
        calibration = []
        for index in range(identity["B_calibration_rows_per_seed"]):
            B_cue = identity["B_cues"][index % len(identity["B_cues"])]
            calibration.append({
                "identity": f"{namespace}:B_calibration:{seed}:{index}",
                "phase": "B_calibration", "seed": seed, "B_cue": B_cue,
                "B_local_target": B_map[B_cue],
                "source_event": _digest(namespace, "B_calibration", seed, index, "source"),
            })
        random.Random(seed + 811).shuffle(calibration)
        rows.extend(calibration)
        training = []
        combinations = itertools.product(
            identity["A_cues"], identity["B_cues"], (0, 1), (0, 1), range(2)
        )
        for index, (A_cue, B_cue, A_action, B_action, repeat) in enumerate(combinations):
            A_target = A_map[A_cue]
            B_target = B_map[B_cue]
            target = A_target ^ B_target
            training.append({
                "identity": f"{namespace}:main_training:{seed}:{index}",
                "phase": "main_training", "seed": seed,
                "A_cue": A_cue, "B_cue": B_cue,
                "forced_A_action": A_action, "forced_B_action": B_action,
                "repeat": repeat, "A_target": A_target,
                "B_local_target": B_target, "global_target": target,
                "global_success": int(B_action == target),
                "source_event": _digest(namespace, "main_training", seed, index, "source"),
                "outcome_id": _digest(namespace, "main_training", seed, index, "outcome"),
            })
        random.Random(seed + 919).shuffle(training)
        rows.extend(training)
        heldout = []
        for index, (A_cue, B_cue, repeat) in enumerate(itertools.product(
            identity["A_cues"], identity["B_cues"], range(2)
        )):
            A_target = A_map[A_cue]
            B_target = B_map[B_cue]
            heldout.append({
                "identity": f"{namespace}:heldout:{seed}:{index}",
                "phase": "heldout", "seed": seed,
                "A_cue": A_cue, "B_cue": B_cue, "repeat": repeat,
                "A_target": A_target, "B_local_target": B_target,
                "global_target": A_target ^ B_target,
                "source_event": _digest(namespace, "heldout", seed, index, "source"),
                "outcome_id": _digest(namespace, "heldout", seed, index, "outcome"),
            })
        random.Random(seed + 1021).shuffle(heldout)
        rows.extend(heldout)
    return rows


def _majority_accuracy(rows: list[dict], feature) -> float:
    groups: dict[object, Counter] = defaultdict(Counter)
    for row in rows:
        groups[feature(row)][row["A_target"]] += 1
    return sum(max(counts.values()) for counts in groups.values()) / len(rows)


def audit(rows: list[dict], protocol: dict) -> dict:
    identity = protocol["identity"]
    by_seed_phase: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_seed_phase[(row["seed"], row["phase"])].append(row)
    training = [row for row in rows if row["phase"] == "main_training"]
    all_ids = [row["identity"] for row in rows]
    source_ids = [row["source_event"] for row in rows]
    outcome_ids = [row["outcome_id"] for row in rows if "outcome_id" in row]
    oracle_agreement = all(
        row["B_local_target"] == _oracle_map(row["seed"], identity["B_cues"], "B-map")[row["B_cue"]]
        and (
            row["phase"] == "B_calibration"
            or (
                row["A_target"] == _oracle_map(row["seed"], identity["A_cues"], "A-map")[row["A_cue"]]
                and row["global_target"] == (
                    _oracle_map(row["seed"], identity["A_cues"], "A-map")[row["A_cue"]]
                    ^ _oracle_map(row["seed"], identity["B_cues"], "B-map")[row["B_cue"]]
                )
            )
        )
        for row in rows
    )
    shortcut_accuracy = _majority_accuracy(
        training,
        lambda row: (row["forced_A_action"], row["forced_B_action"], row["global_success"]),
    )
    outcome_accuracy = _majority_accuracy(training, lambda row: row["global_success"])
    source_bit_accuracy = _majority_accuracy(training, lambda row: int(row["source_event"][-1], 16) & 1)
    outcome_bit_accuracy = _majority_accuracy(training, lambda row: int(row["outcome_id"][-1], 16) & 1)
    checks = {
        "row_count": len(rows) == len(identity["seeds"]) * (128 + 512 + 128),
        "independent_oracle_agreement": oracle_agreement,
        "balanced_maps": all(
            Counter(_oracle_map(seed, cues, tag).values()) == Counter({0: 4, 1: 4})
            for seed in identity["seeds"]
            for cues, tag in ((identity["A_cues"], "A-map"), (identity["B_cues"], "B-map"))
        ),
        "split_counts": all(
            len(by_seed_phase[(seed, phase)]) == count
            for seed in identity["seeds"]
            for phase, count in (("B_calibration", 128), ("main_training", 512), ("heldout", 128))
        ),
        "complete_action_factorial": all(
            Counter((row["forced_A_action"], row["forced_B_action"])
                    for row in by_seed_phase[(seed, "main_training")]
                    if row["A_cue"] == A_cue and row["B_cue"] == B_cue)
            == Counter({(0, 0): 2, (0, 1): 2, (1, 0): 2, (1, 1): 2})
            for seed in identity["seeds"]
            for A_cue in identity["A_cues"]
            for B_cue in identity["B_cues"]
        ),
        "balanced_training_outcomes": all(
            Counter(row["global_success"] for row in by_seed_phase[(seed, "main_training")])
            == Counter({0: 256, 1: 256})
            for seed in identity["seeds"]
        ),
        "balanced_heldout_targets": all(
            Counter(row["global_target"] for row in by_seed_phase[(seed, "heldout")])
            == Counter({0: 64, 1: 64})
            for seed in identity["seeds"]
        ),
        "direct_shortcut_uninformative": shortcut_accuracy <= 0.55,
        "outcome_only_uninformative": outcome_accuracy <= 0.55,
        "source_id_probe": source_bit_accuracy <= 0.55,
        "outcome_id_probe": outcome_bit_accuracy <= 0.55,
        "all_identities_disjoint": len(all_ids) == len(set(all_ids)),
        "all_source_ids_unique": len(source_ids) == len(set(source_ids)),
        "all_outcome_ids_unique": len(outcome_ids) == len(set(outcome_ids)),
        "candidate_source_frozen": all(
            hashlib.sha256((ROOT / protocol["frozen_sources"][f"{name}_path"]).read_bytes()).hexdigest()
            == protocol["frozen_sources"][f"{name}_sha256"]
            for name in ("candidate", "packet")
        ),
    }
    return {
        "checks": checks,
        "diagnostics": {
            "direct_shortcut_A_target_majority_accuracy": shortcut_accuracy,
            "outcome_only_A_target_majority_accuracy": outcome_accuracy,
            "source_id_low_bit_A_target_accuracy": source_bit_accuracy,
            "outcome_id_low_bit_A_target_accuracy": outcome_bit_accuracy,
        },
    }


def main() -> int:
    raw = PROTOCOL.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen held-out protocol changed")
    protocol = json.loads(raw)
    rows = generate(protocol)
    diagnostics = audit(rows, protocol)
    output_rows = Path(ensure_parent_directory(ROWS))
    output_rows.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = {
        "schema": "sara-local-credit-packet-multihop-heldout-materialization-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "rows_sha256": hashlib.sha256(output_rows.read_bytes()).hexdigest(),
        "row_count": len(rows),
        **diagnostics,
        "passed": all(diagnostics["checks"].values()),
        "heldout_materialized": True,
        "heldout_consumed": False,
        "candidate_execution_authorized": all(diagnostics["checks"].values()),
        "production_authorized": False,
    }
    output_report = Path(ensure_parent_directory(workspace_path(
        "evaluation", "local_credit_packet_multihop_heldout_materialization.json"
    )))
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_report), **report}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
