#!/usr/bin/env python3
"""Generate and independently audit mixed-error revision held-out rows."""
from __future__ import annotations

from collections import defaultdict
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
from scripts.eval.local_credit_packet_online_budget_heldout_materialize import (  # noqa: E402
    audit as base_audit, generate,
)

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_mixed_error_revision_heldout_v1.json"
))
PROTOCOL_SHA256 = "e332b8bc5d6626d212bc7e553fc7499bf9a93728265964fef7d5d6f65fbfa638"
PARENT_GENERATOR = ROOT / "scripts/eval/local_credit_packet_online_budget_heldout_materialize.py"
ROWS = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_mixed_error_revision_heldout_rows_v1.jsonl"
))
AUDIT = Path(workspace_path(
    "evaluation", "local_credit_packet_mixed_error_revision_heldout_audit_v1.json"
))


def _ordered(cues: list[int], namespace: str, label: str, seed: int) -> list[int]:
    ranks = {
        cue: int.from_bytes(hashlib.new(
            "sha256", (namespace + "|" + label + "|" + str(seed) + "|" + str(cue)).encode()
        ).digest(), "big") for cue in cues
    }
    return sorted(cues, key=ranks.__getitem__)


def _private_map(seed: int, cues: list[int], tag: str) -> dict[int, int]:
    ranks = {cue: int(hashlib.new(
        "sha256", (str(seed) + "|" + str(cue) + "|" + tag).encode()
    ).hexdigest(), 16) for cue in cues}
    ordered = sorted(cues, key=ranks.__getitem__)
    return {cue: int(ordered.index(cue) >= 4) for cue in cues}


def independent_oracle(rows: list[dict], protocol: dict) -> dict:
    identity = protocol["identity"]
    namespace = identity["namespace"]
    checks = {}
    per_seed = {}
    for seed in identity["seeds"]:
        training = [row for row in rows if row["seed"] == seed and row["phase"] == "training"]
        test = [row for row in rows if row["seed"] == seed and row["phase"] == "test"]
        A_map = _private_map(seed, identity["A_cues"], "A-map")
        B_map = _private_map(seed, identity["B_cues"], "B-map")
        wrong = set(_ordered(identity["B_cues"], namespace, "initial_wrong", seed)[:4])
        visible = set(_ordered(identity["B_cues"], namespace, "revision_visible", seed)[:6])
        corrupt = _ordered(identity["B_cues"], namespace, "revision_corrupt", seed)[0]
        initial = {cue: B_map[cue] ^ int(cue in wrong) for cue in identity["B_cues"]}
        revised = {
            "full": B_map,
            "partial_six": {cue: B_map[cue] if cue in visible else initial[cue]
                            for cue in identity["B_cues"]},
            "one_corrupt": {cue: B_map[cue] ^ int(cue == corrupt)
                            for cue in identity["B_cues"]},
        }
        anchors = []
        seen = defaultdict(int)
        for index, row in enumerate(training):
            cue = row["A_cue"]
            if seen[cue] < 2:
                seen[cue] += 1
                anchors.append((index + 1, row))
        seed_checks = {
            "rows_match_private_oracle": all(
                row["A_target"] == A_map[row["A_cue"]]
                and row["B_local_target"] == B_map[row["B_cue"]]
                and row["global_target"] == (A_map[row["A_cue"]] ^ B_map[row["B_cue"]])
                and (row["phase"] != "training" or row["global_success"]
                     == int(row["forced_B_action"] == row["global_target"]))
                for row in training + test
            ),
            "sixteen_anchors": len(anchors) == 16 and set(seen.values()) == {2},
            "four_wrong_initial": sum(initial[cue] != B_map[cue]
                                      for cue in identity["B_cues"]) == 4,
            "full_revision_exact": revised["full"] == B_map,
            "partial_revision_exact": all(
                revised["partial_six"][cue] == (B_map[cue] if cue in visible else initial[cue])
                for cue in identity["B_cues"]),
            "one_corrupt_exact": sum(revised["one_corrupt"][cue] != B_map[cue]
                                     for cue in identity["B_cues"]) == 1,
        }
        condition_results = {}
        for condition, B_code in revised.items():
            weights = defaultdict(int)
            changed = 0
            for _, row in anchors:
                A_cue, B_cue = row["A_cue"], row["B_cue"]
                A_action, B_action = row["forced_A_action"], row["forced_B_action"]
                desired_B = B_action ^ (1 - row["global_success"])
                old_sign = 1 if A_action == (desired_B ^ initial[B_cue]) else -1
                new_sign = 1 if A_action == (desired_B ^ B_code[B_cue]) else -1
                weights[(A_cue, A_action)] += old_sign
                if old_sign != new_sign:
                    weights[(A_cue, A_action)] += 2 * new_sign
                    changed += 1
            correct = 0
            for row in test:
                A_action = int(weights[(row["A_cue"], 1)] > weights[(row["A_cue"], 0)])
                predicted = A_action ^ B_code[row["B_cue"]]
                correct += int(predicted == row["global_target"])
            condition_results[condition] = {
                "global_accuracy": correct / len(test),
                "changed_signs": changed,
            }
        seed_checks["full_two_step_exact"] = condition_results["full"]["global_accuracy"] == 1.0
        checks[str(seed)] = seed_checks
        per_seed[str(seed)] = {
            "anchor_source_events": [row["source_event"] for _, row in anchors],
            "initially_wrong_B_cues": sorted(wrong),
            "visible_revision_B_cues": sorted(visible),
            "corrupt_revision_B_cue": corrupt,
            "conditions": condition_results,
        }
    return {"checks": checks, "per_seed": per_seed,
            "passed": all(all(group.values()) for group in checks.values())}


def main() -> int:
    if ROWS.exists() or AUDIT.exists():
        raise ValueError("Held-out rows or audit already exist; refusing regeneration")
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if hashlib.sha256(PARENT_GENERATOR.read_bytes()).hexdigest() != protocol["parent_generator_sha256"]:
        raise ValueError("Independent parent generator changed")
    rows = generate(protocol)
    base = base_audit(rows, protocol)
    oracle = independent_oracle(rows, protocol)
    checks = {**base["checks"], "mixed_error_independent_oracle": oracle["passed"]}
    if not all(checks.values()):
        print(json.dumps({"checks": checks, "oracle_checks": oracle["checks"]}, sort_keys=True))
        return 1
    row_bytes = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    result = {
        "schema": "sara-mixed-error-revision-heldout-pre-execution-audit-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "rows_sha256": hashlib.sha256(row_bytes).hexdigest(),
        "row_count": len(rows),
        "direct_shortcut_probe": base["direct_shortcut_probe"],
        "independent_oracle": oracle,
        "checks": checks,
        "passed": True,
        "heldout_consumed": False,
    }
    Path(ensure_parent_directory(ROWS)).write_bytes(row_bytes)
    Path(ensure_parent_directory(AUDIT)).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"rows_sha256": result["rows_sha256"], "row_count": len(rows),
                      "direct_shortcut_probe": result["direct_shortcut_probe"],
                      "passed": result["passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
