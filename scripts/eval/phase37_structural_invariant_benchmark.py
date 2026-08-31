#!/usr/bin/env python3
"""Execute the single registered Phase 37 structural-invariant attempt."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase37_preregistration import MECHANISM_ARMS  # noqa: E402
from sara_engine.risa.structural_invariant import CanonicalTypedMotifStore, MotifEdge  # noqa: E402
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_parent_directory  # noqa: E402

TRAIN = processed_data_path("benchmark_fixtures", "phase37_structural_train_base.jsonl")
INPUTS = processed_data_path("benchmark_fixtures", "phase37_structural_execution_inputs_v2.jsonl")
KEY = processed_data_path("benchmark_fixtures", "phase37_structural_execution_evaluator_key_v2.jsonl")
REGISTRATION = workspace_path("evaluation", "phase37_structural_invariant_preregistration.json")
DEFAULT_REPORT = workspace_path("evaluation", "phase37_structural_invariant_benchmark.json")
EXPECTED = {
    "inputs": "ccfccd4de1a602ca7e69faf20c808165755f7676b6760221f6dd077efe81695d",
    "key": "30fe7274b49c24d5b420f6ce5bf499433ed6e54f8425be3fe4277ac7db2da9ea",
    "registration": "e77d34460bfc2ae2440d765616a65ce7dad734d07ef6cca3b0d17b1532cfe704",
}


def _rows(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _file_sha(path: str) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _edges(rows: Sequence[Mapping[str, Any]]) -> List[MotifEdge]:
    return [MotifEdge(str(x["source"]), str(x["relation_type"]), str(x["target"]), str(x.get("evidence_id", "")), bool(x.get("verified", False))) for x in rows]


def _store(train: Sequence[Mapping[str, Any]]) -> CanonicalTypedMotifStore:
    store = CanonicalTypedMotifStore(max_patterns=64, max_edges_per_pattern=6, max_candidate_patterns=8)
    for row in train:
        store.observe(str(row["case_id"]), _edges(row["visible_edges"]), context=str(row["semantic_domain"]))
    return store


def _candidate(arm: str, store: CanonicalTypedMotifStore, row: Mapping[str, Any]) -> Dict[str, Any]:
    if arm in MECHANISM_ARMS[:3]:
        return {"abstained": True, "proposals": [], "reason": f"{arm}_cannot_generate_unstored_relation", "event_cost": len(row["visible_edges"]), "state_bytes": 0, "trace": {"durable_mutation_allowed": False}}
    result = store.propose(
        _edges(row["visible_edges"]),
        context=str(row["query"]["context"]),
        context_aware=arm != "canonical_typed_motif_context_free",
        shuffled_binding=arm == "intact_candidate_shuffled_binding_control",
        max_proposals=int(row["max_proposals"]),
    )
    return result.to_dict()


def build_report(train: Sequence[Mapping[str, Any]], inputs: Sequence[Mapping[str, Any]], keys: Sequence[Mapping[str, Any]], registration: Mapping[str, Any]) -> Dict[str, Any]:
    if registration.get("protocol_fingerprint") != EXPECTED["registration"]:
        raise ValueError("Phase 37 registration identity mismatch")
    key_by_id = {row["case_id"]: row for row in keys}
    store = _store(train)
    results = []
    arm_metrics: Dict[str, Dict[str, float]] = {}
    for arm in MECHANISM_ARMS:
        tp = fp = fn = correct = abstain_correct = abstain_total = event_cost = max_state = 0
        family_correct: Dict[str, int] = {}
        evidence_complete = evidence_total = role_consistent = role_total = 0
        deterministic = True
        start = time.perf_counter()
        for row in inputs:
            candidate = _candidate(arm, store, row)
            deterministic = deterministic and candidate == _candidate(arm, store, row)
            key = key_by_id[row["case_id"]]
            proposed = bool(candidate["proposals"])
            expected = key["expected_decision"] == "propose"
            relation_match = proposed and any(p["relation_type"] == key["withheld_edge"]["relation_type"] for p in candidate["proposals"])
            tp += int(expected and relation_match)
            fp += int(proposed and not relation_match)
            fn += int(expected and not relation_match)
            correct += int((expected and relation_match) or (not expected and not proposed))
            case_correct = int((expected and relation_match) or (not expected and not proposed))
            family_correct[str(key["case_family"])] = case_correct
            for proposal in candidate["proposals"]:
                evidence_total += 1
                evidence_complete += int(bool(proposal.get("evidence_ids")))
                role_total += 1
                role_consistent += int(proposal.get("source_role") == "role:source" and proposal.get("target_role") == "role:target")
            if not expected:
                abstain_total += 1
                abstain_correct += int(not proposed)
            event_cost += int(candidate["event_cost"])
            max_state = max(max_state, int(candidate["state_bytes"]))
            results.append({"arm": arm, "case_id": row["case_id"], "candidate": candidate, "evaluation": {"correct": bool((expected and relation_match) or (not expected and not proposed)), "relation_match": bool(relation_match)}})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        arm_metrics[arm] = {
            "verified_novel_relation_precision": tp / max(1, tp + fp),
            "verified_novel_relation_recall": tp / max(1, tp + fn),
            "justified_abstention_accuracy": abstain_correct / max(1, abstain_total),
            "decision_accuracy": correct / max(1, len(inputs)),
            "heldout_domain_transfer_accuracy": float(family_correct.get("heldout_domain", 0)),
            "rare_exception_preservation": float(family_correct.get("rare_exception", 0)),
            "direction_order_sensitivity": (family_correct.get("temporal_order_reversal", 0) + family_correct.get("causal_direction_reversal", 0)) / 2.0,
            "role_map_consistency": role_consistent / max(1, role_total),
            "evidence_chain_completeness": evidence_complete / max(1, evidence_total),
            "revision_retraction_accuracy": float(family_correct.get("revised_evidence", 0)),
            "deterministic_replay": float(deterministic),
            "event_cost": float(event_cost),
            "max_state_bytes": float(max_state),
            "cpu_latency_ms": elapsed_ms,
        }
    intact = arm_metrics["canonical_typed_motif_context_exception_aware"]
    thresholds = registration["thresholds"]
    threshold_checks = {name: intact[name] >= float(rule["limit"]) for name, rule in thresholds.items() if name in intact}
    threshold_checks.update({name: False for name in thresholds if name not in intact})
    baseline_best = max(arm_metrics[arm]["decision_accuracy"] for arm in MECHANISM_ARMS[:4])
    comparative = intact["decision_accuracy"] >= baseline_best + float(registration["comparative_acceptance"]["minimum_quality_lift_over_each_existing_baseline"])
    passed = bool(all(threshold_checks.values()) and comparative)
    return {
        "schema": "sara-phase37-structural-invariant-benchmark-v1",
        "protocol_fingerprint": EXPECTED["registration"],
        "single_registered_attempt_consumed": True,
        "passed": passed,
        "promotion_ready": False,
        "retained_negative_result": not passed,
        "arm_metrics": arm_metrics,
        "checks": {"all_registered_thresholds_passed": all(threshold_checks.values()), "intact_beats_all_baselines": comparative, "all_proposals_provisional": all(not p.get("durable_mutation_allowed", True) for item in results for p in item["candidate"].get("proposals", [])), "resource_bounds_passed": all(m["event_cost"] <= registration["budgets"]["max_event_cost"] and m["max_state_bytes"] <= registration["budgets"]["max_state_bytes"] for m in arm_metrics.values())},
        "threshold_checks": threshold_checks,
        "results": results,
        "claim_boundary": "The first frozen attempt is reported without retuning. Failure is retained and does not authorize fixture changes, production mutation, or a generalization claim.",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    if _file_sha(INPUTS) != EXPECTED["inputs"] or _file_sha(KEY) != EXPECTED["key"]:
        print("Phase 37 frozen execution identity mismatch", file=sys.stderr)
        return 2
    with open(REGISTRATION, encoding="utf-8") as handle:
        registration = json.load(handle)
    report = build_report(_rows(TRAIN), _rows(INPUTS), _rows(KEY), registration)
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"passed": report["passed"], "retained_negative_result": report["retained_negative_result"], "checks": report["checks"], "intact": report["arm_metrics"]["canonical_typed_motif_context_exception_aware"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
