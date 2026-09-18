#!/usr/bin/env python3
"""Independently materialize non-oracle anchor held-out identities before scoring."""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_allowed_output_path, processed_data_path, workspace_path,
)

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_nonoracle_anchor_acquisition_heldout_v1.json"
))
PROTOCOL_SHA256 = "6afee9eb95988e03a38af7af1870abc0bb7bf3888c86d75fa9a560a636d0afa1"
ROWS = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_nonoracle_anchor_acquisition_heldout_rows_v1.jsonl"
))
AUDIT = Path(workspace_path(
    "evaluation", "local_credit_packet_nonoracle_anchor_acquisition_heldout_audit_v1.json"
))
PARENT_PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_nonoracle_anchor_acquisition_v1.json"
))
PARENT_RESULT = Path(workspace_path(
    "evaluation", "local_credit_packet_nonoracle_anchor_acquisition_development_v1.json"
))
CANDIDATE = ROOT / "scripts/eval/local_credit_packet_nonoracle_anchor_acquisition.py"


def _digest(*parts: object) -> bytes:
    return sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()


def _private_map(seed: int, cues: list[int], tag: str) -> dict[int, int]:
    ranks = {cue: int.from_bytes(_digest(seed, cue, tag), "big") for cue in cues}
    order = sorted(cues, key=ranks.__getitem__)
    return {cue: int(position >= len(cues) // 2)
            for position, cue in enumerate(order)}


def _oracle_map(seed: int, cues: list[int], tag: str) -> dict[int, int]:
    ranks = {cue: int(sha256(f"{seed}|{cue}|{tag}".encode("utf-8")).hexdigest(), 16)
             for cue in cues}
    ordered = sorted(ranks, key=lambda cue: ranks[cue])
    return {cue: int(ordered.index(cue) >= len(cues) // 2) for cue in cues}


def _oracle_arm(selected: list[tuple[int, dict]], A_map: dict[int, int],
                B_map: dict[int, int], wrong: set[int], arm: str) -> dict:
    weights: dict[tuple[int, int], int] = defaultdict(int)
    changed_by_cue = Counter()
    packets = 0
    for _, row in selected:
        cue, B_cue, branch = row["A_cue"], row["B_cue"], row["A_action"]
        old_target = A_map[cue] ^ int(B_cue in wrong)
        old_sign = 1 if branch == old_target else -1
        weights[(cue, branch)] += old_sign
    if arm != "prior_only":
        for _, row in selected:
            cue, B_cue, branch = row["A_cue"], row["B_cue"], row["A_action"]
            changed = B_cue in wrong
            changed_by_cue[cue] += int(changed)
            if not changed and arm != "unconditional_two_step":
                continue
            sign = 1 if branch == A_map[cue] else -1
            if arm == "sign_shuffle_two_step":
                sign = -sign
            target_branch = branch ^ int(arm == "route_shuffle_two_step")
            magnitude = 1 if arm.startswith("simple") else 2
            weights[(cue, target_branch)] += magnitude * sign
            packets += 1
    else:
        for _, row in selected:
            changed_by_cue[row["A_cue"]] += int(row["B_cue"] in wrong)
    tie_value = int(arm == "simple_tie_one")
    per_cue = {}
    for cue, target in A_map.items():
        zero, one = weights[(cue, 0)], weights[(cue, 1)]
        prediction = tie_value if zero == one else int(one > zero)
        per_cue[str(cue)] = {"target": target, "changed": changed_by_cue[cue],
                             "prediction": prediction, "correct": prediction == target,
                             "weights": [zero, one]}
    return {"per_cue": per_cue, "packets": packets}


def generate(protocol: dict) -> list[dict]:
    identity = protocol["identity"]
    namespace = identity["namespace"]
    rows = []
    for seed in identity["seeds"]:
        A_map = _private_map(seed, identity["A_cues"], "A-map")
        B_map = _private_map(seed, identity["B_cues"], "B-map")
        training = []
        for original_index, (A_cue, B_cue, A_action, B_action) in enumerate(
            itertools.product(identity["A_cues"], identity["B_cues"], (0, 1), (0, 1))
        ):
            training.append({
                "identity": f"{namespace}:training:{seed}:{original_index}",
                "phase": "training", "seed": seed, "original_index": original_index,
                "A_cue": A_cue, "B_cue": B_cue,
                "A_action": A_action, "B_action": B_action,
                "success": int(B_action == (A_map[A_cue] ^ B_map[B_cue])),
                "source_event": _digest(namespace, "source", seed, original_index).hex()[:24],
            })
        training.sort(key=lambda row: _digest(namespace, "order", seed,
                                              row["original_index"]))
        rows.extend(training)
        for test_index, (A_cue, B_cue) in enumerate(itertools.product(
            identity["A_cues"], identity["B_cues"]
        )):
            rows.append({
                "identity": f"{namespace}:test:{seed}:{test_index}",
                "phase": "test", "seed": seed, "test_index": test_index,
                "A_cue": A_cue, "B_cue": B_cue,
                "A_target": A_map[A_cue], "B_local_target": B_map[B_cue],
            })
    return rows


def independent_audit(rows: list[dict], protocol: dict) -> dict:
    identity = protocol["identity"]
    namespace = identity["namespace"]
    checks: dict[str, bool] = {}
    strata = Counter()
    first_two = {}
    success_counts = Counter()
    for seed in identity["seeds"]:
        training = [row for row in rows if row["seed"] == seed
                    and row["phase"] == "training"]
        test = [row for row in rows if row["seed"] == seed and row["phase"] == "test"]
        A_map = _oracle_map(seed, identity["A_cues"], "A-map")
        B_map = _oracle_map(seed, identity["B_cues"], "B-map")
        wrong_ranks = {cue: int.from_bytes(
            _digest(namespace, "initial_wrong", seed, cue), "big")
            for cue in identity["B_cues"]}
        wrong = set(sorted(wrong_ranks, key=wrong_ranks.__getitem__)[:4])
        seen = defaultdict(int)
        selected = []
        for position, row in enumerate(training, 1):
            cue = row["A_cue"]
            if seen[cue] < 2:
                seen[cue] += 1
                selected.append((position, row))
            success_counts[(A_map[cue], row["success"])] += 1
        for cue in identity["A_cues"]:
            chosen = [(position, row["original_index"], row["B_cue"])
                      for position, row in selected if row["A_cue"] == cue]
            first_two[f"{seed}:{cue}"] = chosen
            changed = sum(row["B_cue"] in wrong for _, row in selected
                          if row["A_cue"] == cue)
            strata[(changed, A_map[cue])] += 1
        training_factorial = set(itertools.product(
            identity["A_cues"], identity["B_cues"], (0, 1), (0, 1)))
        test_factorial = set(itertools.product(identity["A_cues"], identity["B_cues"]))
        checks[f"{seed}:balanced_maps"] = (
            Counter(A_map.values()) == {0: 8, 1: 8}
            and Counter(B_map.values()) == {0: 4, 1: 4}
            and A_map == _private_map(seed, identity["A_cues"], "A-map")
            and B_map == _private_map(seed, identity["B_cues"], "B-map"))
        checks[f"{seed}:factorials"] = (
            len(training) == len(training_factorial) == 512
            and len(test) == len(test_factorial) == 128
            and {tuple(row[field] for field in
                       ("A_cue", "B_cue", "A_action", "B_action"))
                 for row in training} == training_factorial
            and {(row["A_cue"], row["B_cue"]) for row in test} == test_factorial)
        checks[f"{seed}:order_and_outcomes"] = (
            [row["original_index"] for row in training]
            == sorted(range(512), key=lambda index: _digest(namespace, "order", seed, index))
            and all(row["success"] == int(row["B_action"]
                        == (A_map[row["A_cue"]] ^ B_map[row["B_cue"]]))
                    for row in training)
            and all(row["A_target"] == A_map[row["A_cue"]]
                    and row["B_local_target"] == B_map[row["B_cue"]]
                    for row in test))
        checks[f"{seed}:first_two"] = (
            len(selected) == 32 and set(seen.values()) == {2}
            and len(wrong) == 4
            and all(len(first_two[f"{seed}:{cue}"]) == 2 for cue in identity["A_cues"]))
        arm_oracles = {arm: _oracle_arm(selected, A_map, B_map, wrong, arm)
                       for arm in protocol["evaluation"]["arms"]}
        checks[f"{seed}:oracle_two_step"] = all(
            all(row["correct"] for row in arm_oracles[arm]["per_cue"].values())
            for arm in ("selective_two_step", "unconditional_two_step"))
        checks[f"{seed}:oracle_packet_counts"] = (
            arm_oracles["selective_two_step"]["packets"]
            == sum(row["B_cue"] in wrong for _, row in selected)
            and arm_oracles["unconditional_two_step"]["packets"] == 32)
    checks["identity_uniqueness"] = (len({row["identity"] for row in rows}) == len(rows)
                                    and len({row["source_event"] for row in rows
                                             if row["phase"] == "training"})
                                    == 512 * len(identity["seeds"]))
    checks["direct_success_balanced"] = all(
        success_counts[(target, success)] == 128 * len(identity["seeds"])
        for target in (0, 1) for success in (0, 1))
    checks["two_change_both_targets"] = all(strata[(2, target)] > 0
                                              for target in (0, 1))
    return {
        "checks": checks, "passed": all(checks.values()),
        "changed_sign_strata": {str(changed): {str(target): strata[(changed, target)]
                                            for target in (0, 1)} for changed in range(3)},
        "first_two_identities": first_two,
        "rows": len(rows), "candidate_scored": False,
    }


def main() -> int:
    if sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Held-out protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    for path, key in ((PARENT_PROTOCOL, "parent_development_protocol_sha256"),
                      (PARENT_RESULT, "parent_development_result_sha256"),
                      (CANDIDATE, "candidate_source_sha256")):
        if sha256(path.read_bytes()).hexdigest() != protocol[key]:
            raise ValueError(f"Pinned source changed: {path.name}")
    rows_path = Path(ensure_allowed_output_path(str(ROWS)))
    audit_path = Path(ensure_allowed_output_path(str(AUDIT)))
    if rows_path.exists() or audit_path.exists():
        raise ValueError("Held-out materialization already exists")
    rows = generate(protocol)
    audit = independent_audit(rows, protocol)
    if not audit["passed"]:
        raise ValueError("Pre-execution held-out audit failed")
    row_bytes = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"))
                           + "\n").encode("utf-8") for row in rows)
    audit["protocol_sha256"] = PROTOCOL_SHA256
    audit["rows_sha256"] = sha256(row_bytes).hexdigest()
    audit_bytes = (json.dumps(audit, sort_keys=True, separators=(",", ":"))
                   + "\n").encode("utf-8")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("xb") as stream:
        stream.write(row_bytes)
    with audit_path.open("xb") as stream:
        stream.write(audit_bytes)
    print(json.dumps({"rows_sha256": audit["rows_sha256"],
                      "audit_sha256": sha256(audit_bytes).hexdigest(),
                      "changed_sign_strata": audit["changed_sign_strata"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
