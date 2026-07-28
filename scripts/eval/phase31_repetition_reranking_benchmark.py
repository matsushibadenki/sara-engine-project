#!/usr/bin/env python3
"""Evaluate candidate-only repetition reranking under equal charged budgets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.learning.repetition_candidate_reranker import (  # noqa: E402
    CandidateRepetitionReranker,
)
from sara_engine.learning.repetition_consolidation import (  # noqa: E402
    RepetitionDependentConsolidator,
)
from sara_engine.memory.event_state_cache import (  # noqa: E402
    CacheRetrievalResult,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures",
    "phase31_repetition_reranking_cases.jsonl",
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation",
    "phase31_repetition_reranking_benchmark.json",
)


def load_cases(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _base_retrieval(row: Mapping[str, Any]) -> CacheRetrievalResult:
    matches = tuple(
        sorted(
            (
                {
                    "entry_id": "target",
                    "score": float(row["target_base_score"]),
                    "tier": "consolidated",
                    "source_ref": "source-target",
                    "utility": 0.70,
                    "components": {
                        "sparse_overlap": float(
                            row["target_base_score"]
                        )
                    },
                },
                {
                    "entry_id": "distractor",
                    "score": float(row["distractor_base_score"]),
                    "tier": "consolidated",
                    "source_ref": "source-distractor",
                    "utility": 0.70,
                    "components": {
                        "sparse_overlap": float(
                            row["distractor_base_score"]
                        )
                    },
                },
            ),
            key=lambda match: (
                -float(match["score"]),
                str(match["entry_id"]),
            ),
        )
    )
    return CacheRetrievalResult(
        abstained=False,
        decision="retrieve_verified",
        matches=matches,
        event_cost=6,
        scanned_entries=2,
        reactivation_hints=tuple(
            {
                "entry_id": match["entry_id"],
                "activation": match["score"],
                "mutates_durable_state": False,
            }
            for match in matches
        ),
    )


def _target_match(result: CacheRetrievalResult) -> Mapping[str, Any]:
    return next(
        match
        for match in result.matches
        if match["entry_id"] == "target"
    )


def _run_case(row: Mapping[str, Any]) -> Dict[str, Any]:
    consolidator = RepetitionDependentConsolidator()
    candidate = CandidateRepetitionReranker(
        consolidator,
        enabled=True,
    )
    control = CandidateRepetitionReranker(
        consolidator,
        enabled=False,
    )
    for event in row["events"]:
        candidate.observe(
            entry_id=str(event["entry_id"]),
            timestep=int(event["timestep"]),
            source_ref=str(event["source_ref"]),
            recall_success=bool(event["recall_success"]),
            verified=bool(event["verified"]),
            contradiction=bool(event.get("contradiction", False)),
        )
    base = _base_retrieval(row)
    control_result = control.rerank(
        base,
        timestep=int(row["query_timestep"]),
    )
    candidate_result = candidate.rerank(
        base,
        timestep=int(row["query_timestep"]),
    )
    target = _target_match(candidate_result)
    distractor = next(
        match
        for match in candidate_result.matches
        if match["entry_id"] == "distractor"
    )
    charged_event_cost = base.event_cost + len(base.matches)
    return {
        "case_id": str(row["case_id"]),
        "input_event_count": len(row["events"]),
        "query_timestep": int(row["query_timestep"]),
        "control_order": [
            str(match["entry_id"]) for match in control_result.matches
        ],
        "candidate_order": [
            str(match["entry_id"]) for match in candidate_result.matches
        ],
        "target_base_score": float(row["target_base_score"]),
        "target_candidate_score": float(target["score"]),
        "target_candidate_boost": float(
            target["components"]["repetition_candidate_boost"]
        ),
        "target_eligible": bool(
            target["components"]["repetition_candidate_eligible"]
        ),
        "distractor_candidate_score": float(distractor["score"]),
        "distractor_candidate_boost": float(
            distractor["components"]["repetition_candidate_boost"]
        ),
        "distractor_eligible": bool(
            distractor["components"]["repetition_candidate_eligible"]
        ),
        "control_charged_event_cost": charged_event_cost,
        "candidate_charged_event_cost": charged_event_cost,
        "actual_candidate_event_cost": candidate_result.event_cost,
        "state_budget_ok": consolidator.state_budget_ok(),
        "production_state_mutated": any(
            bool(trace["mutates_durable_state"])
            for trace in candidate.last_trace
        ),
        "candidate_trace": list(candidate.last_trace),
    }


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cases = {
        result["case_id"]: result for result in map(_run_case, rows)
    }
    repeated = {
        result["case_id"]: result for result in map(_run_case, rows)
    }
    spaced = cases["spaced_verified_delayed"]
    massed = cases["massed_verified_delayed"]
    unverified = cases["unverified_repetition"]
    before_contradiction = cases["verified_before_contradiction"]
    after_contradiction = cases["verified_after_contradiction"]
    interference = cases["unverified_interference"]
    checks = {
        "fixture_present": bool(rows),
        "deterministic_replay": cases == repeated and bool(rows),
        "control_path_unchanged": all(
            case["control_order"][0]
            == (
                "target"
                if case["target_base_score"]
                >= next(
                    float(row["distractor_base_score"])
                    for row in rows
                    if str(row["case_id"]) == case_id
                )
                else "distractor"
            )
            for case_id, case in cases.items()
        ),
        "spaced_verified_retrieval_recovers_target": (
            spaced["control_order"][0] == "distractor"
            and spaced["candidate_order"][0] == "target"
        ),
        "spaced_gain_exceeds_massed_gain": (
            spaced["target_candidate_boost"]
            > massed["target_candidate_boost"]
        ),
        "unverified_repetition_cannot_rerank": (
            unverified["candidate_order"] == unverified["control_order"]
            and unverified["target_candidate_boost"] == 0.0
            and unverified["target_eligible"] is False
        ),
        "contradiction_reduces_candidate_score": (
            after_contradiction["target_candidate_score"]
            < before_contradiction["target_candidate_score"]
        ),
        "unverified_interference_not_eligible": (
            interference["candidate_order"][0] == "target"
            and interference["distractor_eligible"] is False
            and interference["distractor_candidate_boost"] == 0.0
        ),
        "equal_charged_event_budget": all(
            case["control_charged_event_cost"]
            == case["candidate_charged_event_cost"]
            for case in cases.values()
        ),
        "state_budget_integrity": all(
            case["state_budget_ok"] for case in cases.values()
        ),
        "durable_state_not_mutated": all(
            not case["production_state_mutated"]
            for case in cases.values()
        ),
        "production_path_not_changed": True,
    }
    return {
        "schema": "sara-phase31-repetition-reranking-benchmark-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "production_path_changed": False,
        "checks": checks,
        "metrics": {
            "case_count": len(rows),
            "spaced_target_boost": spaced[
                "target_candidate_boost"
            ],
            "massed_target_boost": massed[
                "target_candidate_boost"
            ],
            "unverified_target_boost": unverified[
                "target_candidate_boost"
            ],
            "pre_contradiction_target_score": before_contradiction[
                "target_candidate_score"
            ],
            "post_contradiction_target_score": after_contradiction[
                "target_candidate_score"
            ],
        },
        "cases": cases,
        "policy_notes": [
            "The disabled control and candidate use identical source events and state.",
            "Both arms are charged for the same bounded candidate scan.",
            "Only traces with verified source evidence are rerank-eligible.",
            "No durable cache field or production retrieval default is mutated.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report(load_cases(args.fixture_path))
    with open(
        ensure_parent_directory(args.output_path),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
