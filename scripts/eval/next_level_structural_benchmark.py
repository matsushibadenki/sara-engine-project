#!/usr/bin/env python3
"""Run the observed-only Phase 21 bounded structural reasoning benchmark."""

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

from sara_engine.risa.subgraph_reasoning import (  # noqa: E402
    BoundedSubgraphComposer,
    StructuralAnalogyEngine,
    SubgraphEdge,
)
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402

DEFAULT_FIXTURE = processed_data_path("benchmark_fixtures", "next_level_structural_cases.jsonl")
DEFAULT_REPORT = workspace_path("evaluation", "next_level_structural_benchmark.json")
DEFAULT_SUMMARY = workspace_path("evaluation", "next_level_structural_benchmark_summary.txt")


def _edges(rows: Sequence[Mapping[str, Any]]) -> List[SubgraphEdge]:
    return [
        SubgraphEdge(
            source=str(row["source"]),
            target=str(row["target"]),
            relation_type=str(row["relation_type"]),
            confidence=float(row.get("confidence", 0.0)),
            evidence_count=int(row.get("evidence_count", 0)),
            context_tags=tuple(str(item) for item in row.get("context_tags", []) or []),
            verified=bool(row.get("verified", False)),
        )
        for row in rows
    ]


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    supported_ok = unsupported_ok = analogy_ok = analogy_abstain_ok = True
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if "edges" in row:
            result = BoundedSubgraphComposer(max_hops=3, max_paths=4).compose(
                _edges(row["edges"]),
                source=str(row["source"]),
                target=str(row["target"]),
                context_tags=tuple(str(item) for item in row.get("context_tags", []) or []),
            )
            results[case_id] = result.to_dict()
            if case_id == "composition_supported":
                supported_ok = len(result.proposals) == 1 and result.abstained is False
            if case_id == "composition_unsupported":
                unsupported_ok = result.abstained is True
        else:
            result = StructuralAnalogyEngine(min_score=0.5).compare(_edges(row["left_edges"]), _edges(row["right_edges"]))
            results[case_id] = result.to_dict()
            if case_id == "analogy_supported":
                analogy_ok = result.score == 1.0 and result.abstained is False
            if case_id == "analogy_unsupported":
                analogy_abstain_ok = result.abstained is True
    passed = bool(supported_ok and unsupported_ok and analogy_ok and analogy_abstain_ok)
    return {
        "schema": "sara-next-level-structural-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "metrics": {
            "supported_composition": float(supported_ok),
            "unsupported_composition_abstention": float(unsupported_ok),
            "supported_structural_analogy": float(analogy_ok),
            "unsupported_structural_analogy_abstention": float(analogy_abstain_ok),
            "durable_mutation_boundary": float(
                all(
                    all(not proposal.get("durable_mutation_allowed", True) for proposal in item.get("proposals", []))
                    for item in results.values()
                    if isinstance(item, Mapping) and "proposals" in item
                )
            ),
        },
        "cases": results,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    with open(args.fixture_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    report = build_report(rows)
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(ensure_parent_directory(args.summary_path), "w", encoding="utf-8") as handle:
        handle.write(f"Next-level structural benchmark: {'PASS' if report['passed'] else 'FAIL'}\n")
        handle.write(f"Observed only: {report['observed_only']}\n")
        for key, value in report["metrics"].items():
            handle.write(f"- {key}: {value}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
