#!/usr/bin/env python3
"""Run the observed-only RISA structural interpolation benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.risa.structural_interpolation import (  # noqa: E402
    PredictiveStructuralFeedbackEngine,
    StructuralEvidence,
    StructuralFeedbackSignal,
    StructuralInterpolationEngine,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)


DEFAULT_FIXTURE = processed_data_path("benchmark_fixtures", "structural_interpolation_cases.jsonl")
DEFAULT_REPORT = workspace_path("evaluation", "structural_interpolation_benchmark.json")
DEFAULT_SUMMARY = workspace_path("evaluation", "structural_interpolation_benchmark_summary.txt")


def _load(path: str) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(payload)
    return rows


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[StructuralEvidence]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("case_id", "unknown"))].append(
            StructuralEvidence(
                source_node=str(row.get("source_node", "")),
                target_node=str(row.get("target_node", "")),
                relation_type=str(row.get("relation_type", "related")),
                confidence=float(row.get("confidence", 0.0)),
                source_ref=str(row.get("source_ref", "")),
                source_hash=str(row.get("source_hash", "")),
                source_revision=str(row.get("source_revision", "")),
                context_tags=tuple(str(item) for item in row.get("context_tags", []) or []),
                acquired_at=int(row.get("acquired_at", 0)),
                contradiction_count=int(row.get("contradiction_count", 0)),
                expiry_segment=row.get("expiry_segment"),
                metabolic_cost=int(row.get("metabolic_cost", 0)),
                verified=bool(row.get("verified", True)),
            )
        )
    case_reports = {}
    for case_id, evidence in sorted(grouped.items()):
        result = StructuralInterpolationEngine(max_proposals=8).propose(evidence, current_segment=100)
        case_reports[case_id] = result.to_dict()
    merge = case_reports.get("independent_merge", {})
    duplicate = case_reports.get("same_source_duplicate", {})
    contradiction = case_reports.get("contradiction_block", {})
    revision = case_reports.get("source_revision_recovery", {})
    context = case_reports.get("context_separation", {})
    merge_ok = len(merge.get("proposals", [])) == 1
    duplicate_ok = len(duplicate.get("proposals", [])) == 0
    contradiction_ok = len(contradiction.get("proposals", [])) == 0
    revision_proposals = revision.get("proposals", [])
    revision_ok = len(revision_proposals) == 1 and set(revision_proposals[0].get("source_revisions", [])) == {"r1", "r2"}
    context_ok = len(context.get("proposals", [])) == 0 and context.get("trace", {}).get("group_count") == 2
    feedback_engine = PredictiveStructuralFeedbackEngine()
    unsupported = feedback_engine.propose(
        (
            StructuralFeedbackSignal(
                predicting_concept="concept:animal",
                source_node="concept:unknown",
                target_node="concept:mammal",
                relation_type="instance_of",
                predicted_confidence=0.8,
                observed_confidence=0.0,
                evidence_ids=(),
                eligible=False,
                rollback_state="verified_snapshot",
            ),
        )
    )[0]
    oscillation = feedback_engine.propose(
        (
            StructuralFeedbackSignal(
                predicting_concept="concept:animal",
                source_node="concept:dog",
                target_node="concept:mammal",
                relation_type="instance_of",
                predicted_confidence=0.5,
                observed_confidence=0.5,
                evidence_ids=("e-1",),
                recent_actions=("strengthen_relation", "cut_relation", "strengthen_relation", "cut_relation"),
                rollback_state="verified_snapshot",
            ),
        )
    )[0]
    unsupported_ok = unsupported.edit_type == "request_more_evidence"
    rollback_ok = oscillation.edit_type == "freeze_subgraph" and oscillation.rollback_state == "verified_snapshot"
    passed = bool(merge_ok and duplicate_ok and contradiction_ok and revision_ok and context_ok and unsupported_ok and rollback_ok)
    return {
        "schema": "sara-structural-interpolation-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "case_count": len(case_reports),
        "metrics": {
            "independent_merge_proposal": float(merge_ok),
            "same_source_duplicate_block": float(duplicate_ok),
            "contradiction_block": float(contradiction_ok),
            "durable_mutation_block": float(
                all(not item.get("durable_mutation_allowed", True) for item in merge.get("proposals", []))
            ),
            "source_revision_recovery": float(revision_ok),
            "context_separation": float(context_ok),
            "unsupported_neighbor_abstention": float(unsupported_ok),
            "oscillation_rollback_freeze": float(rollback_ok),
        },
        "cases": case_reports,
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
        handle.write(f"Structural interpolation benchmark: {'PASS' if report['passed'] else 'FAIL'}\n")
        handle.write(f"Observed only: {report['observed_only']}\n")
        for key, value in report["metrics"].items():
            handle.write(f"- {key}: {value}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
