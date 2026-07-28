#!/usr/bin/env python3
"""Run the observed-only Phase 24 causal and counterfactual benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.risa.causal_reasoning import (  # noqa: E402
    BoundedCausalReasoner,
    CausalEvidence,
    causal_event_state_candidate,
)
from sara_engine.memory.event_state_cache import VerifiedHierarchicalEventStateCache  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402

DEFAULT_FIXTURE = processed_data_path("benchmark_fixtures", "phase24_causal_cases.jsonl")
DEFAULT_REPORT = workspace_path("evaluation", "phase24_causal_benchmark.json")
DEFAULT_SUMMARY = workspace_path("evaluation", "phase24_causal_benchmark_summary.txt")


def _load(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    reasoner = BoundedCausalReasoner(max_branch_depth=3)
    cases: Dict[str, Any] = {}
    memory_cache = VerifiedHierarchicalEventStateCache(max_entries=4)
    for row in rows:
        evidence = tuple(CausalEvidence(**item) for item in row.get("evidence", []))
        inference = reasoner.infer(evidence)
        branch = reasoner.counterfactual(
            inference,
            intervention="remove_cause",
            predicted_outcome="target_present",
            alternative_outcome="target_absent",
        )
        branch_result = reasoner.branch_counterfactual(
            inference,
            intervention="remove_cause",
            predicted_outcome="target_present",
            alternative_outcome="target_absent",
            context_tags=(f"case:{row['case_id']}",),
        )
        rollback = reasoner.rollback_counterfactual(
            branch_result,
            reason="benchmark_transaction_cleanup",
        )
        admission = memory_cache.admit(
            causal_event_state_candidate(
                inference,
                source_ref=f"fixture:{row['case_id']}",
                time_segment=len(cases),
            )
        )
        cases[str(row["case_id"])] = {
            "expected_relation": str(row["expected_relation"]),
            "counterfactual_expected_abstain": bool(row["counterfactual_expected_abstain"]),
            "inference": inference.to_dict(),
            "counterfactual": branch,
            "counterfactual_branch_result": branch_result.to_dict(),
            "counterfactual_rollback": rollback.to_dict(),
            "event_memory_admission": admission.to_dict(),
        }
    checks = {
        "relation_classification": all(item["inference"]["relation_type"] == item["expected_relation"] for item in cases.values()),
        "temporal_order_not_verified_causal": cases["temporal_only"]["inference"]["relation_type"] == "causes_candidate",
        "source_conflict_abstention": cases["source_conflict"]["inference"]["abstained"] is True,
        "unstable_feedback_freeze": (
            cases["unstable_feedback"]["inference"]["reason"]
            == "unstable_feedback_freeze"
        ),
        "unsupported_counterfactual_abstention": all(
            item["counterfactual"]["abstained"] == item["counterfactual_expected_abstain"]
            for item in cases.values()
        ),
        "verified_branch_bounded": cases["intervention_verified"]["counterfactual"]["branch_count"] == 2,
        "branch_records_bounded_and_traceable": (
            cases["intervention_verified"]["counterfactual_branch_result"]["branch_count"] == 2
            and cases["intervention_verified"]["counterfactual_branch_result"]["depth"] <= 3
            and cases["intervention_verified"]["counterfactual_branch_result"]["event_cost"] <= 16
            and cases["intervention_verified"]["counterfactual_branch_result"]["serialized_state_bytes"] <= 4096
            and all(
                item["supporting_paths"]
                and "room:a" in item["context_tags"]
                and item["durable_mutation_allowed"] is False
                for item in cases["intervention_verified"]["counterfactual_branch_result"]["branches"]
            )
        ),
        "explicit_rollback_isolated": (
            cases["intervention_verified"]["counterfactual_branch_result"]["rolled_back"] is False
            and cases["intervention_verified"]["counterfactual_rollback"]["rolled_back"] is True
            and all(
                item["status"] == "rolled_back"
                for item in cases["intervention_verified"]["counterfactual_rollback"]["branches"]
            )
        ),
        "support_paths_and_alternatives_present": all(
            item["inference"]["supporting_paths"]
            and item["inference"]["alternatives"]
            for item in cases.values()
        ),
        "durable_mutation_blocked": all(
            not item["inference"]["durable_mutation_allowed"]
            and not item["counterfactual"]["durable_mutation_allowed"]
            for item in cases.values()
        ),
        "event_memory_only_verified_causal": (
            cases["intervention_verified"]["event_memory_admission"]["accepted"]
            and all(
                not cases[case_id]["event_memory_admission"]["accepted"]
                for case_id in (
                    "temporal_only",
                    "source_conflict",
                    "unsupported_counterfactual",
                    "unstable_feedback",
                )
            )
        ),
    }
    return {
        "schema": "sara-phase24-causal-benchmark-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "external_device_required": False,
        "metrics": {
            "case_count": len(cases),
            "verified_causal_case": float(checks["verified_branch_bounded"]),
            "causal_event_memory_admission": float(checks["event_memory_only_verified_causal"]),
            "counterfactual_branch_record_integrity": float(
                checks["branch_records_bounded_and_traceable"]
                and checks["explicit_rollback_isolated"]
            ),
        },
        "checks": checks,
        "cases": cases,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    report = build_report(_load(args.fixture_path))
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(ensure_parent_directory(args.summary_path), "w", encoding="utf-8") as handle:
        handle.write(f"Phase 24 causal benchmark: {'PASS' if report['passed'] else 'FAIL'}\n")
        for key, value in report["metrics"].items():
            handle.write(f"- {key}: {value}\n")
        for key, value in report["checks"].items():
            handle.write(f"- check.{key}: {value}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
