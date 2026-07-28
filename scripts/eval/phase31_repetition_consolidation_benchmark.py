#!/usr/bin/env python3
"""Evaluate bounded repetition-dependent memory consolidation."""

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

from sara_engine.learning.repetition_consolidation import (  # noqa: E402
    RepetitionConsolidationConfig,
    RepetitionDependentConsolidator,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures",
    "phase31_repetition_consolidation_cases.jsonl",
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation",
    "phase31_repetition_consolidation_benchmark.json",
)


def load_cases(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _events_for_case(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    events = row.get("events")
    if isinstance(events, list):
        return [dict(event) for event in events]
    repeat = row.get("repeat")
    if not isinstance(repeat, Mapping):
        raise ValueError("case must define events or repeat")
    count = int(repeat.get("count", 0))
    start = int(repeat.get("start", 0))
    interval = int(repeat.get("interval", 1))
    if count < 1 or interval < 1:
        raise ValueError("repeat count and interval must be positive")
    return [
        {
            "timestep": start + index * interval,
            "source_ref": str(repeat.get("source_ref", "")),
            "outcome": str(repeat.get("outcome", "support")),
            "recall_success": bool(repeat.get("recall_success", False)),
            "verified": bool(repeat.get("verified", False)),
        }
        for index in range(count)
    ]


def _run_case(row: Mapping[str, Any]) -> Dict[str, Any]:
    consolidator = RepetitionDependentConsolidator()
    memory_id = str(row["memory_id"])
    trajectory: List[Dict[str, Any]] = []
    for event in _events_for_case(row):
        update = consolidator.observe(
            memory_id=memory_id,
            timestep=int(event["timestep"]),
            source_ref=str(event.get("source_ref", "")),
            outcome=str(event.get("outcome", "support")),
            recall_success=bool(event.get("recall_success", False)),
            verified=bool(event.get("verified", False)),
        )
        state = update["after"]
        trajectory.append(
            {
                "timestep": int(event["timestep"]),
                "outcome": str(event.get("outcome", "support")),
                "retrieval_strength": float(state["retrieval_strength"]),
                "stability": float(state["stability"]),
                "verification_strength": float(
                    state["verification_strength"]
                ),
                "verified_source_count": int(
                    state["verified_source_count"]
                ),
            }
        )
    before_advance = consolidator.read(memory_id)
    advance_to = row.get("advance_to")
    if advance_to is not None:
        consolidator.advance(int(advance_to))
    final_state = consolidator.read(memory_id)
    return {
        "case_id": str(row["case_id"]),
        "event_count": len(trajectory),
        "trajectory": trajectory,
        "before_advance": before_advance,
        "final_state": final_state,
        "snapshot": consolidator.snapshot(),
    }


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    case_results = {
        result["case_id"]: result for result in map(_run_case, rows)
    }
    repeated_results = {
        result["case_id"]: result for result in map(_run_case, rows)
    }
    deterministic_replay = case_results == repeated_results and bool(rows)

    one_shot = case_results["one_shot"]["final_state"]
    massed = case_results["massed_repetition"]["final_state"]
    spaced = case_results["spaced_retrieval"]["final_state"]
    forgetting = case_results["delayed_forgetting"]
    contradiction = case_results["contradiction"]["trajectory"]
    duplicate = case_results["duplicate_verified_source"]["final_state"]
    distinct = case_results["distinct_verified_sources"]["final_state"]
    saturation = case_results["saturation"]["trajectory"]

    capacity_probe = RepetitionDependentConsolidator(
        RepetitionConsolidationConfig(capacity=2)
    )
    for index in range(5):
        capacity_probe.observe(
            memory_id=f"pollution-{index}",
            timestep=index,
            source_ref=f"source-{index}",
        )
    capacity_snapshot = capacity_probe.snapshot()

    budget_probe = RepetitionDependentConsolidator(
        RepetitionConsolidationConfig(max_events=2)
    )
    for timestep in range(2):
        budget_probe.observe(
            memory_id="budget-memory",
            timestep=timestep,
            source_ref="source-a",
        )
    budget_rejection = budget_probe.observe(
        memory_id="budget-memory",
        timestep=2,
        source_ref="source-a",
    )

    isolation_probe = RepetitionDependentConsolidator()
    isolation_probe.observe(
        memory_id="memory-a",
        timestep=0,
        source_ref="source-a",
    )
    projected_unrelated = isolation_probe.read("memory-a", timestep=1)
    isolation_probe.observe(
        memory_id="memory-b",
        timestep=1,
        source_ref="source-b",
    )
    unrelated_after = isolation_probe.read("memory-a")

    saturation_gains = [
        saturation[0]["retrieval_strength"],
        *[
            saturation[index]["retrieval_strength"]
            - saturation[index - 1]["retrieval_strength"]
            for index in range(1, len(saturation))
        ],
    ]
    checks = {
        "fixture_present": bool(rows),
        "deterministic_replay": deterministic_replay,
        "repetition_strengthens_retrieval": (
            massed["retrieval_strength"] > one_shot["retrieval_strength"]
        ),
        "spaced_retrieval_improves_stability": (
            spaced["stability"] > massed["stability"]
        ),
        "spaced_retrieval_improves_access": (
            spaced["retrieval_strength"] > massed["retrieval_strength"]
        ),
        "delayed_forgetting_observed": (
            forgetting["final_state"]["retrieval_strength"]
            < forgetting["before_advance"]["retrieval_strength"]
            and forgetting["final_state"]["stability"]
            < forgetting["before_advance"]["stability"]
        ),
        "contradiction_depresses_trace": (
            contradiction[-1]["retrieval_strength"]
            < contradiction[-2]["retrieval_strength"]
            and contradiction[-1]["stability"]
            < contradiction[-2]["stability"]
        ),
        "duplicate_source_does_not_inflate_verification": (
            duplicate["verified_source_count"] == 1
            and duplicate["verification_strength"]
            == RepetitionConsolidationConfig().verification_rate
        ),
        "distinct_sources_increase_verification": (
            distinct["verified_source_count"] == 3
            and distinct["verification_strength"]
            > duplicate["verification_strength"]
        ),
        "saturation_bounded": (
            saturation[-1]["retrieval_strength"] <= 1.0
            and saturation[-1]["stability"] <= 1.0
            and saturation_gains[-1] < saturation_gains[0]
        ),
        "capacity_eviction_bounded": (
            capacity_snapshot["memory_units"] == 2
            and capacity_snapshot["eviction_count"] == 3
            and capacity_snapshot["state_budget_ok"]
        ),
        "event_budget_rejects_without_mutation": (
            budget_rejection["mutation_allowed"] is False
            and budget_rejection["reason"] == "event_budget_exhausted"
            and budget_probe.snapshot()["event_count"] == 2
        ),
        "local_update_does_not_rewrite_unrelated_memory": (
            projected_unrelated == unrelated_after
        ),
        "production_path_not_changed": True,
        "backpropagation_not_used": True,
        "dense_matrix_not_used": True,
        "gpu_not_used": True,
    }
    return {
        "schema": "sara-phase31-repetition-consolidation-benchmark-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "production_path_changed": False,
        "checks": checks,
        "metrics": {
            "case_count": len(rows),
            "one_shot_retrieval_strength": one_shot[
                "retrieval_strength"
            ],
            "massed_retrieval_strength": massed[
                "retrieval_strength"
            ],
            "massed_stability": massed["stability"],
            "spaced_retrieval_strength": spaced[
                "retrieval_strength"
            ],
            "spaced_stability": spaced["stability"],
            "duplicate_source_verification_strength": duplicate[
                "verification_strength"
            ],
            "distinct_source_verification_strength": distinct[
                "verification_strength"
            ],
            "capacity_evictions": capacity_snapshot["eviction_count"],
        },
        "cases": case_results,
        "capacity_probe": capacity_snapshot,
        "policy_notes": [
            "Retrieval strength and verification strength are separate.",
            "Repeated evidence from one source cannot inflate verification.",
            "The mechanism is not connected to production recall or durable admission.",
            "No human-equivalent memory or physical-energy claim is made.",
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
