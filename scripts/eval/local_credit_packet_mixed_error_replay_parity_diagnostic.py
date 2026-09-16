#!/usr/bin/env python3
"""Explain one-step versus two-step replay using development rows only."""
from __future__ import annotations

from collections import Counter, defaultdict
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
from scripts.eval.local_credit_packet_noisy_late_B_evidence_v2 import (  # noqa: E402
    audit_selection, materialize, schedule,
)

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_mixed_error_replay_parity_diagnostic_v1.json"
))
PROTOCOL_SHA256 = "b7138bd252c5ce8f09ff3a5e88e1a63393916653d53ce2b4b58473034d2272c2"
DEVELOPMENT_PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_mixed_error_revision_v1.json"
))
DEVELOPMENT_RESULT = Path(workspace_path(
    "evaluation", "local_credit_packet_mixed_error_revision_development_v1.json"
))
SELECTOR = ROOT / "scripts/eval/local_credit_packet_noisy_late_B_evidence_v2.py"
SELECTOR_PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_noisy_late_B_evidence_v1.json"
))
OUTPUT = Path(workspace_path(
    "evaluation", "local_credit_packet_mixed_error_replay_parity_diagnostic_v1.json"
))


def _predict(weights: dict[tuple[int, int], int], cue: int) -> int:
    return int(weights[(cue, 1)] > weights[(cue, 0)])


def analyze_seed(development: dict, selector: dict, seed: int) -> dict:
    rows = materialize(selector, seed)
    training, flips = schedule(selector, seed, rows)
    audit_selection(selector, seed, rows, training, flips)
    namespace = development["identity"]["namespace"]
    B_map = rows["B_map"]
    wrong = set(sorted(B_map, key=lambda cue: hashlib.sha256(
        f"{namespace}|initial_wrong|{seed}|{cue}".encode()
    ).hexdigest())[:4])
    initial = {cue: B_map[cue] ^ int(cue in wrong) for cue in B_map}
    anchors = defaultdict(list)
    for index, event in enumerate(training):
        A_cue = event[0]
        if len(anchors[A_cue]) < 2:
            anchors[A_cue].append((index + 1, event))
    if set(anchors) != set(development["identity"]["A_cues"]) or any(
        len(events) != 2 for events in anchors.values()
    ):
        raise ValueError("Development anchors are incomplete")
    per_cue = {}
    weights = {phase: defaultdict(int) for phase in ("prior", "simple", "correction")}
    for cue in development["identity"]["A_cues"]:
        changed = 0
        for source, (A_cue, B_cue, A_action, B_action, success) in anchors[cue]:
            desired_B = B_action ^ (1 - success)
            old_sign = 1 if A_action == (desired_B ^ initial[B_cue]) else -1
            revised_sign = 1 if A_action == (desired_B ^ B_map[B_cue]) else -1
            for phase in weights:
                weights[phase][(A_cue, A_action)] += old_sign
            if old_sign != revised_sign:
                changed += 1
                weights["simple"][(A_cue, A_action)] += revised_sign
                weights["correction"][(A_cue, A_action)] += 2 * revised_sign
        margins = {phase: value[(cue, 1)] - value[(cue, 0)]
                   for phase, value in weights.items()}
        target = rows["A_map"][cue]
        prior_correct = _predict(weights["prior"], cue) == target
        simple_correct = _predict(weights["simple"], cue) == target
        correction_correct = _predict(weights["correction"], cue) == target
        per_cue[str(cue)] = {
            "anchor_sources": [source for source, _ in anchors[cue]],
            "target": target,
            "changed_sign_count": changed,
            "old_margin": margins["prior"],
            "simple_margin": margins["simple"],
            "correction_margin": margins["correction"],
            "prior_correct": prior_correct,
            "simple_correct": simple_correct,
            "correction_correct": correction_correct,
            "simple_tie": margins["simple"] == 0,
            "second_step_necessary": correction_correct and not simple_correct,
        }
    return {"per_cue": per_cue,
            "accuracy": {phase: sum(_predict(value, cue) == rows["A_map"][cue]
                                    for cue in rows["A_map"]) / len(rows["A_map"])
                         for phase, value in weights.items()}}


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Diagnostic result already exists; refusing regeneration")
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Diagnostic protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    for path, key in ((DEVELOPMENT_PROTOCOL, "development_protocol_sha256"),
                      (DEVELOPMENT_RESULT, "development_result_sha256"),
                      (SELECTOR, "selector_source_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[key]:
            raise ValueError(f"Pinned development input changed: {path.name}")
    development = json.loads(DEVELOPMENT_PROTOCOL.read_text())
    frozen = json.loads(DEVELOPMENT_RESULT.read_text())
    selector = json.loads(SELECTOR_PROTOCOL.read_text())
    selector["identity"] = development["identity"]
    per_seed = {str(seed): analyze_seed(development, selector, seed)
                for seed in development["identity"]["seeds"]}
    consistency = {
        "matches_frozen_accuracy": all(
            per_seed[str(seed)]["accuracy"][phase] == frozen["per_seed"][str(seed)][
                "conditions"]["full"]["arms"][arm]["A_accuracy"]
            for seed in development["identity"]["seeds"]
            for phase, arm in (("prior", "prior_credit_only"),
                               ("simple", "simple_replay"),
                               ("correction", "correction_packet"))),
        "two_anchors_per_cue": all(len(cue["anchor_sources"]) == 2
                                   for seed in per_seed.values()
                                   for cue in seed["per_cue"].values()),
        "correction_all_cues_correct": all(cue["correction_correct"]
                                           for seed in per_seed.values()
                                           for cue in seed["per_cue"].values()),
    }
    all_cues = [cue for seed in per_seed.values() for cue in seed["per_cue"].values()]
    summary = {
        "cue_count": len(all_cues),
        "changed_sign_histogram": dict(sorted(Counter(
            cue["changed_sign_count"] for cue in all_cues
        ).items())),
        "simple_correct_by_changed_count": {str(count): sum(
            cue["simple_correct"] for cue in all_cues if cue["changed_sign_count"] == count
        ) for count in range(3)},
        "second_step_necessary_by_changed_count": {str(count): sum(
            cue["second_step_necessary"] for cue in all_cues if cue["changed_sign_count"] == count
        ) for count in range(3)},
        "simple_tie_correct_count": sum(cue["simple_tie"] and cue["simple_correct"]
                                        for cue in all_cues),
        "simple_tie_wrong_count": sum(cue["simple_tie"] and not cue["simple_correct"]
                                      for cue in all_cues),
    }
    report = {
        "schema": "sara-local-credit-packet-mixed-error-replay-parity-diagnostic-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "summary": summary,
        "consistency_checks": consistency,
        "passed": all(consistency.values()),
        "per_seed": per_seed,
        "heldout_inputs_used": False,
        "exploratory_only": True,
        "production_authorized": False,
    }
    Path(ensure_parent_directory(OUTPUT)).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "consistency_checks": consistency}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
