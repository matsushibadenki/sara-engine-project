#!/usr/bin/env python3
"""Run the observed-only Phase 21 bounded structural reasoning benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from sara_engine.risa.graph_store import RisaGraphStore  # noqa: E402
from sara_engine.risa.models import ConceptCell  # noqa: E402
from sara_engine.risa.structural_edit_transaction import (  # noqa: E402
    BoundedStructuralEditTransaction,
    graph_digest,
)
from sara_engine.risa.structural_interpolation import (  # noqa: E402
    PredictiveStructuralFeedbackEngine,
    StructuralFeedbackSignal,
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

    store = RisaGraphStore()
    store.add_or_update_node(
        ConceptCell(cell_id="concept:animal", kind="concept", label="animal")
    )
    original_digest = graph_digest(store)
    feedback = PredictiveStructuralFeedbackEngine()
    create_proposal = feedback.propose(
        (
            StructuralFeedbackSignal(
                predicting_concept="concept:animal",
                source_node="concept:animal",
                target_node="concept:unknown-animal",
                relation_type="predicts",
                predicted_confidence=0.2,
                observed_confidence=0.8,
                evidence_ids=("phase21-evidence-a", "phase21-evidence-b"),
                context_tags=("biology",),
                target_exists=False,
                provisional_node_kind="animal_candidate",
                provisional_node_label="unknown animal",
            ),
        )
    )[0]
    link_proposal = feedback.propose(
        (
            StructuralFeedbackSignal(
                predicting_concept="concept:animal",
                source_node="concept:animal",
                target_node="concept:unknown-animal",
                relation_type="predicts",
                predicted_confidence=0.2,
                observed_confidence=0.8,
                evidence_ids=("phase21-evidence-a", "phase21-evidence-b"),
                context_tags=("biology",),
            ),
        )
    )[0]
    transaction = BoundedStructuralEditTransaction(max_edits=4)
    staged = transaction.stage(store, (create_proposal, link_proposal))
    rollback = transaction.stage(
        store,
        (
            create_proposal,
            replace(
                link_proposal,
                proposal_id="phase21-invalid-late-edit",
                source_node="concept:missing-source",
            ),
        ),
    )
    provisional_ok = bool(
        create_proposal.edit_type == "create_provisional_node"
        and not create_proposal.durable_mutation_allowed
    )
    staging_ok = bool(
        staged.accepted_for_review
        and staged.staged_edit_count == 2
        and graph_digest(store) == original_digest
        and store.get_node("concept:unknown-animal") is None
    )
    rollback_ok = bool(
        rollback.rolled_back
        and rollback.rollback_verified
        and rollback.final_digest == original_digest
        and rollback.staged_edit_count == 1
    )
    results["provisional_node_batch"] = staged.to_dict()
    results["multi_edit_atomic_rollback"] = rollback.to_dict()
    passed = bool(
        supported_ok
        and unsupported_ok
        and analogy_ok
        and analogy_abstain_ok
        and provisional_ok
        and staging_ok
        and rollback_ok
    )
    return {
        "schema": "sara-next-level-structural-benchmark-v1",
        "passed": passed,
        "observed_only": True,
        "metrics": {
            "supported_composition": float(supported_ok),
            "unsupported_composition_abstention": float(unsupported_ok),
            "supported_structural_analogy": float(analogy_ok),
            "unsupported_structural_analogy_abstention": float(analogy_abstain_ok),
            "provisional_node_boundary": float(provisional_ok),
            "multi_edit_staging_boundary": float(staging_ok),
            "multi_edit_atomic_rollback": float(rollback_ok),
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
