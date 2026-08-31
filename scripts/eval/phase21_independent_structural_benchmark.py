#!/usr/bin/env python3
"""Evaluate Phase 21 on provenance-bound external held-out structures."""

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

from sara_engine.risa.subgraph_reasoning import BoundedSubgraphComposer, StructuralAnalogyEngine, SubgraphEdge  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402

DEFAULT_FIXTURE = processed_data_path("benchmark_fixtures", "phase21_independent_structural_cases.jsonl")
DEFAULT_REPORT = workspace_path("evaluation", "phase21_independent_structural_benchmark.json")
LEGACY_FIXTURE = processed_data_path("benchmark_fixtures", "next_level_structural_cases.jsonl")
ALLOWED_EVIDENCE = {
    "arch-migration-ietf-001": "970c1c462797f6474aaa8258ffd1ccf99564f12bfbef1b8f6214974e53d3f146",
    "arch-migration-ietf-002": "773cbe44a6e46692a91ebaefd007c37f62070956b93805078f5284af0e3b6982",
    "arch-migration-ietf-003": "6403bd1044f6a26d057ba07e0c2dc75e65689269c30855feb9a5d3e50b025e09",
    "arch-migration-python-001": "bf6440403ffa7818c3662954bfa9cc440e1f970be6843f8c06c4660184bb4bc7",
    "arch-migration-python-002": "043ea21e518e9db2cf1cc07ae907de2189c8ee970fe5ded4bd44a036111c27fc",
    "arch-migration-python-003": "352c7e4dc929b68d47484c4f0acfedfbfcb4dc0e91b2ab6941e1aa741121af25",
}


def _edges(rows: Sequence[Mapping[str, Any]]) -> List[SubgraphEdge]:
    return [SubgraphEdge(source=str(x["source"]), target=str(x["target"]), relation_type=str(x["relation_type"]), confidence=float(x["confidence"]), evidence_count=int(x.get("evidence_count", 0)), context_tags=tuple(x.get("context_tags", ())), verified=bool(x.get("verified", False))) for x in rows]


def _single_edge_hit(edges: Sequence[SubgraphEdge], source: str, target: str) -> bool:
    return any(edge.verified and edge.source == source and edge.target == target for edge in edges)


def _legacy_entities() -> set[str]:
    entities: set[str] = set()
    with open(LEGACY_FIXTURE, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for group in (row.get("edges", ()), row.get("left_edges", ()), row.get("right_edges", ())):
                for edge in group:
                    entities.update((edge["source"], edge["target"]))
    return entities


def _provenance_valid(row: Mapping[str, Any]) -> bool:
    evidence = row.get("evidence", {})
    ids = evidence.get("record_ids", ())
    hashes = evidence.get("source_hashes", ())
    return bool(evidence.get("human_reviewed") and ids and len(ids) == len(hashes) and all(ALLOWED_EVIDENCE.get(i) == h for i, h in zip(ids, hashes)))


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    legacy = _legacy_entities()
    results = []
    composition_total = composition_correct = baseline_correct = 0
    abstention_total = abstention_correct = 0
    analogy_total = analogy_correct = 0
    provenance_ok = True
    disjoint_ok = True
    bounded_ok = True
    for row in rows:
        provenance_ok = provenance_ok and _provenance_valid(row)
        raw_edges = row.get("edges", ()) or (*row.get("left_edges", ()), *row.get("right_edges", ()))
        disjoint_ok = disjoint_ok and not any(edge["source"] in legacy or edge["target"] in legacy for edge in raw_edges)
        family = row["family"]
        if family.startswith("composition"):
            edges = _edges(row["edges"])
            result = BoundedSubgraphComposer(max_hops=3, max_paths=4).compose(edges, source=row["source"], target=row["target"], context_tags=row.get("context_tags", ()))
            baseline = _single_edge_hit(edges, row["source"], row["target"])
            expected_supported = family == "composition_supported"
            correct = (not result.abstained) == expected_supported
            if expected_supported:
                composition_total += 1
                composition_correct += int(correct)
                baseline_correct += int(baseline)
            else:
                abstention_total += 1
                abstention_correct += int(result.abstained)
            bounded_ok = bounded_ok and result.event_cost <= len(edges) + 12 and len(result.proposals) <= 4 and all(not p.durable_mutation_allowed for p in result.proposals)
            results.append({"case_id": row["case_id"], "family": family, "correct": correct, "single_edge_hit": baseline, "candidate": result.to_dict(), "evidence": row["evidence"]})
        else:
            result = StructuralAnalogyEngine(min_score=0.5, max_edges=8).compare(_edges(row["left_edges"]), _edges(row["right_edges"]))
            expected_supported = family == "analogy_supported"
            correct = (not result.abstained) == expected_supported
            analogy_total += 1
            analogy_correct += int(correct)
            bounded_ok = bounded_ok and result.compared_edge_count <= 16
            results.append({"case_id": row["case_id"], "family": family, "correct": correct, "candidate": result.to_dict(), "evidence": row["evidence"]})
    composition_accuracy = composition_correct / max(1, composition_total)
    baseline_accuracy = baseline_correct / max(1, composition_total)
    metrics = {
        "case_count": len(rows),
        "supported_composition_accuracy": composition_accuracy,
        "single_edge_supported_accuracy": baseline_accuracy,
        "composition_lift_over_single_edge": composition_accuracy - baseline_accuracy,
        "unsupported_abstention_accuracy": abstention_correct / max(1, abstention_total),
        "analogy_decision_accuracy": analogy_correct / max(1, analogy_total),
    }
    checks = {
        "external_provenance_bound": provenance_ok,
        "legacy_fixture_entity_disjoint": disjoint_ok,
        "bounded_observed_only": bounded_ok,
        "composition_improves_over_single_edge": composition_accuracy > baseline_accuracy,
        "all_decisions_correct": all(item["correct"] for item in results),
    }
    return {
        "schema": "sara-phase21-independent-structural-benchmark-v1",
        "passed": all(checks.values()),
        "promotion_ready": False,
        "claim_boundary": "External evidence is independently sourced and human reviewed; structural edge decomposition is benchmark-authored, not autonomously learned.",
        "metrics": metrics,
        "checks": checks,
        "results": results,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    with open(args.fixture_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    report = build_report(rows)
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
