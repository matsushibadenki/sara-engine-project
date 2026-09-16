#!/usr/bin/env python3
"""Materialize and independently audit frozen prior-error repair held-out rows."""
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
from scripts.eval.local_credit_packet_online_budget_heldout_materialize import (  # noqa: E402
    audit as base_audit, generate,
)

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_prior_error_repair_heldout_v1.json"
))
PROTOCOL_SHA256 = "b56539b08a950b60c352ae64ad0c8484317971a6a02695569f7faa6b143f2170"
PARENT_GENERATOR = ROOT / "scripts/eval/local_credit_packet_online_budget_heldout_materialize.py"
ROWS = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_prior_error_repair_heldout_rows_v1.jsonl"
))
AUDIT = Path(workspace_path(
    "evaluation", "local_credit_packet_prior_error_repair_heldout_audit_v1.json"
))


def repair_oracle(rows: list[dict], protocol: dict) -> dict:
    seeds = protocol["identity"]["seeds"]
    first_anchors = {}
    sign_opposites = 0
    wrong_predictions = 0
    corrected_predictions = 0
    for seed in seeds:
        training = [row for row in rows if row["seed"] == seed and row["phase"] == "training"]
        anchors = {}
        for row in training:
            anchors.setdefault(row["A_cue"], row)
        first_anchors[str(seed)] = [row["source_event"] for row in anchors.values()]
        for cue, row in anchors.items():
            B_target = int(row["B_local_target"])
            A_target = int(row["A_target"])
            A_action = int(row["forced_A_action"])
            B_action = int(row["forced_B_action"])
            success = int(row["global_success"])
            desired_B_action = B_action ^ (1 - success)
            wrong_desired_A = desired_B_action ^ (1 - B_target)
            corrected_desired_A = desired_B_action ^ B_target
            prior_sign = 1 if A_action == wrong_desired_A else -1
            corrected_sign = 1 if A_action == corrected_desired_A else -1
            sign_opposites += int(prior_sign == -corrected_sign)
            wrong_weight = {0: 0, 1: 0}
            wrong_weight[A_action] += prior_sign
            wrong_prediction = int(wrong_weight[1] > wrong_weight[0])
            wrong_predictions += int(wrong_prediction == A_target)
            corrected_weight = wrong_weight.copy()
            corrected_weight[A_action] += 2 * corrected_sign
            corrected_prediction = int(corrected_weight[1] > corrected_weight[0])
            corrected_predictions += int(corrected_prediction == A_target)
    total = len(seeds) * 8
    checks = {
        "eight_anchor_cues_per_seed": all(len(sources) == 8 for sources in first_anchors.values()),
        "all_prior_signs_reverse": sign_opposites == total,
        "all_prior_predictions_wrong": wrong_predictions == 0,
        "independent_two_step_repair_exact": corrected_predictions == total,
    }
    return {"checks": checks, "first_anchor_sources": first_anchors,
            "opposite_sign_count": sign_opposites,
            "wrong_prediction_accuracy": wrong_predictions / total,
            "corrected_prediction_accuracy": corrected_predictions / total}


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
    repair = repair_oracle(rows, protocol)
    checks = {**base["checks"], **repair["checks"]}
    if not all(checks.values()):
        print(json.dumps({"checks": checks}, sort_keys=True))
        return 1
    row_bytes = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    result = {
        "schema": "sara-prior-error-repair-heldout-pre-execution-audit-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "rows_sha256": hashlib.sha256(row_bytes).hexdigest(),
        "row_count": len(rows),
        "direct_shortcut_probe": base["direct_shortcut_probe"],
        "repair_oracle": repair,
        "checks": checks,
        "passed": True,
        "heldout_consumed": False,
    }
    rows_path = Path(ensure_parent_directory(ROWS))
    audit_path = Path(ensure_parent_directory(AUDIT))
    rows_path.write_bytes(row_bytes)
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": str(rows_path), "audit": str(audit_path),
                      "rows_sha256": result["rows_sha256"],
                      "row_count": len(rows), "passed": result["passed"],
                      "direct_shortcut_probe": result["direct_shortcut_probe"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
