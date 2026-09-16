#!/usr/bin/env python3
"""Independently generate and audit online-budget held-out rows before scoring."""
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
    ensure_parent_directory, processed_data_path, workspace_path,
)

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_online_budget_heldout_v1.json"
))
PROTOCOL_SHA256 = "0f49b9b131a88a6d50b516e4a9d01df39b7d36b7170ff118c86329c4d2bd3d8c"
ROWS = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_online_budget_heldout_rows_v1.jsonl"
))
AUDIT = Path(workspace_path(
    "evaluation", "local_credit_packet_online_budget_heldout_audit_v1.json"
))


def _id(namespace: str, phase: str, seed: int, index: int, kind: str) -> str:
    return hashlib.sha256(
        f"{namespace}|{phase}|{seed}|{index}|{kind}".encode()
    ).hexdigest()[:24]


def _map(seed: int, cues: list[int], tag: str) -> dict[int, int]:
    order = sorted(cues, key=lambda cue: hashlib.sha256(
        f"{seed}|{cue}|{tag}".encode()
    ).digest())
    return {cue: int(position >= 4) for position, cue in enumerate(order)}


def _oracle(seed: int, cues: list[int], tag: str) -> dict[int, int]:
    ranks = {cue: int(hashlib.new(
        "sha256", (str(seed) + "|" + str(cue) + "|" + tag).encode()
    ).hexdigest(), 16) for cue in cues}
    order = sorted(ranks, key=ranks.__getitem__)
    return {cue: int(order.index(cue) >= 4) for cue in cues}


def generate(protocol: dict) -> list[dict]:
    identity = protocol["identity"]
    namespace = identity["namespace"]
    rows = []
    for seed in identity["seeds"]:
        A_map = _map(seed, identity["A_cues"], "A-map")
        B_map = _map(seed, identity["B_cues"], "B-map")
        training = []
        for index, (A_cue, B_cue, A_action, B_action, repeat) in enumerate(
            itertools.product(identity["A_cues"], identity["B_cues"],
                              (0, 1), (0, 1), range(2))
        ):
            A_target = A_map[A_cue]
            B_target = B_map[B_cue]
            training.append({
                "identity": f"{namespace}:training:{seed}:{index}",
                "phase": "training", "seed": seed, "repeat": repeat,
                "A_cue": A_cue, "B_cue": B_cue,
                "forced_A_action": A_action, "forced_B_action": B_action,
                "A_target": A_target, "B_local_target": B_target,
                "global_target": A_target ^ B_target,
                "global_success": int(B_action == (A_target ^ B_target)),
                "source_event": _id(namespace, "training", seed, index, "source"),
                "outcome_id": _id(namespace, "training", seed, index, "outcome"),
            })
        random.Random(seed + 2111).shuffle(training)
        rows.extend(training)
        test = []
        for index, (A_cue, B_cue, repeat) in enumerate(itertools.product(
            identity["A_cues"], identity["B_cues"], range(2)
        )):
            A_target = A_map[A_cue]
            B_target = B_map[B_cue]
            test.append({
                "identity": f"{namespace}:test:{seed}:{index}",
                "phase": "test", "seed": seed, "repeat": repeat,
                "A_cue": A_cue, "B_cue": B_cue,
                "A_target": A_target, "B_local_target": B_target,
                "global_target": A_target ^ B_target,
                "source_event": _id(namespace, "test", seed, index, "source"),
                "outcome_id": _id(namespace, "test", seed, index, "outcome"),
            })
        random.Random(seed + 2221).shuffle(test)
        rows.extend(test)
    return rows


def _majority_accuracy(rows: list[dict], feature) -> float:
    groups: dict[object, Counter] = defaultdict(Counter)
    for row in rows:
        groups[feature(row)][row["A_target"]] += 1
    return sum(max(counts.values()) for counts in groups.values()) / len(rows)


def audit(rows: list[dict], protocol: dict) -> dict:
    identity = protocol["identity"]
    by_seed_phase = defaultdict(list)
    for row in rows:
        by_seed_phase[(row["seed"], row["phase"])].append(row)
    training = [row for row in rows if row["phase"] == "training"]
    ids = [row["identity"] for row in rows]
    sources = [row["source_event"] for row in rows]
    outcomes = [row["outcome_id"] for row in rows]
    oracle_agreement = all(
        row["A_target"] == _oracle(row["seed"], identity["A_cues"], "A-map")[row["A_cue"]]
        and row["B_local_target"] == _oracle(row["seed"], identity["B_cues"], "B-map")[row["B_cue"]]
        and row["global_target"] == (row["A_target"] ^ row["B_local_target"])
        and (row["phase"] != "training" or row["global_success"]
             == int(row["forced_B_action"] == row["global_target"]))
        for row in rows
    )
    direct_probe = _majority_accuracy(
        training, lambda row: (row["forced_A_action"], row["forced_B_action"],
                               row["global_success"])
    )
    checks = {
        "row_count": len(rows) == 3200,
        "independent_oracle_agreement": oracle_agreement,
        "balanced_private_maps": all(
            Counter(_oracle(seed, cues, tag).values()) == Counter({0: 4, 1: 4})
            for seed in identity["seeds"]
            for cues, tag in ((identity["A_cues"], "A-map"),
                              (identity["B_cues"], "B-map"))
        ),
        "split_counts": all(
            len(by_seed_phase[(seed, phase)]) == count
            for seed in identity["seeds"]
            for phase, count in (("training", 512), ("test", 128))
        ),
        "complete_action_factorial": all(
            Counter((row["forced_A_action"], row["forced_B_action"])
                    for row in by_seed_phase[(seed, "training")]
                    if row["A_cue"] == A_cue and row["B_cue"] == B_cue)
            == Counter({(0, 0): 2, (0, 1): 2, (1, 0): 2, (1, 1): 2})
            for seed in identity["seeds"]
            for A_cue in identity["A_cues"]
            for B_cue in identity["B_cues"]
        ),
        "balanced_training_outcomes": all(
            Counter(row["global_success"] for row in by_seed_phase[(seed, "training")])
            == Counter({0: 256, 1: 256}) for seed in identity["seeds"]
        ),
        "balanced_test_targets": all(
            Counter(row["global_target"] for row in by_seed_phase[(seed, "test")])
            == Counter({0: 64, 1: 64}) for seed in identity["seeds"]
        ),
        "direct_shortcut_uninformative": direct_probe <= 0.55,
        "opaque_ids_unique": len(ids) == len(set(ids))
        and len(sources) == len(set(sources))
        and len(outcomes) == len(set(outcomes)),
        "train_test_id_disjoint": set(row["identity"] for row in training).isdisjoint(
            row["identity"] for row in rows if row["phase"] == "test"
        ),
        "anchor_coverage_before_first_teaching": all(
            {row["A_cue"] for row in by_seed_phase[(seed, "training")][:510]}
            == set(identity["A_cues"]) for seed in identity["seeds"]
        ),
    }
    return {"schema": "sara-online-budget-heldout-pre-execution-audit-v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "row_count": len(rows), "direct_shortcut_probe": direct_probe,
            "checks": checks, "passed": all(checks.values()),
            "heldout_consumed": False}


def main() -> int:
    if ROWS.exists() or AUDIT.exists():
        raise ValueError("Held-out rows or audit already exist; refusing regeneration")
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    rows = generate(protocol)
    result = audit(rows, protocol)
    if not result["passed"]:
        print(json.dumps(result, sort_keys=True))
        return 1
    row_bytes = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    result["rows_sha256"] = hashlib.sha256(row_bytes).hexdigest()
    rows_path = Path(ensure_parent_directory(ROWS))
    audit_path = Path(ensure_parent_directory(AUDIT))
    rows_path.write_bytes(row_bytes)
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": str(rows_path), "audit": str(audit_path),
                      "rows_sha256": result["rows_sha256"],
                      "passed": result["passed"],
                      "direct_shortcut_probe": result["direct_shortcut_probe"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
