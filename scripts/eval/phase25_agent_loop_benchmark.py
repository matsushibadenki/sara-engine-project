#!/usr/bin/env python3
"""Run the observed-only Phase 25 bounded agent-loop benchmark."""

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

from sara_engine.agent.bounded_agent_loop import BoundedAgentLoop  # noqa: E402
from sara_engine.agent.transactional_tools import (  # noqa: E402
    BoundedTransactionalToolAdapter,
    ToolStateEdit,
    TransactionalToolRequest,
)
from sara_engine.memory.event_state_cache import VerifiedHierarchicalEventStateCache  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402

DEFAULT_FIXTURE = processed_data_path("benchmark_fixtures", "phase25_agent_cases.jsonl")
DEFAULT_REPORT = workspace_path("evaluation", "phase25_agent_loop_benchmark.json")
DEFAULT_SUMMARY = workspace_path("evaluation", "phase25_agent_loop_benchmark_summary.txt")


def _load(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    loop = BoundedAgentLoop(max_risk=0.75)
    cache = VerifiedHierarchicalEventStateCache(max_entries=4)
    cases: Dict[str, Any] = {}
    for row in rows:
        decision = loop.evaluate_plan(
            goal=str(row["goal"]),
            structural_prediction=str(row.get("structural_prediction", "")),
            expected_outcome=str(row.get("expected_outcome", "")),
            rollback_action=str(row.get("rollback_action", "")),
            risk=float(row.get("risk", 1.0)),
            plan_case=row["plan_case"],
            active_goal=str(row.get("active_goal", "")),
        )
        observation_evidence = {
            "case_id": str(row["case_id"]),
            "observed_outcome": str(row.get("observed_outcome", "")),
        }
        outcome = loop.verify_outcome(
            decision,
            observed_outcome=str(row.get("observed_outcome", "")),
            observation_verified=True,
            observation_evidence=observation_evidence,
        )
        candidate = loop.outcome_event_state_candidate(
            decision,
            observed_outcome=str(row.get("observed_outcome", "")),
            source_ref=f"fixture:{row['case_id']}",
            observation_verified=True,
            observation_evidence=observation_evidence,
        )
        admission = cache.admit(candidate).to_dict() if candidate is not None else None
        action_selection_case = row.get("action_selection", {})
        action_selection = None
        if isinstance(action_selection_case, Mapping) and action_selection_case:
            action_selection = loop.compare_action_selection(
                candidates=action_selection_case.get("candidates", ()),
                structural_feedback=action_selection_case.get(
                    "structural_feedback", ()
                ),
                event_budget_per_arm=int(
                    action_selection_case.get("event_budget_per_arm", 1)
                ),
            ).to_dict()
        tool_commit = None
        tool_rollback = None
        if str(row["case_id"]) == "safe_plan":
            adapter = BoundedTransactionalToolAdapter(
                allowed_tools=("bounded_state_edit",),
                max_edits=4,
                max_event_cost=8,
                max_state_bytes=1024,
            )
            request = TransactionalToolRequest(
                request_id="phase25-safe-plan-tool",
                tool_name="bounded_state_edit",
                goal=decision.goal,
                expected_outcome=decision.expected_outcome,
                rollback_action=decision.rollback_action,
                source_ref="fixture:safe-plan-tool",
                edits=(
                    ToolStateEdit("set", "door_state", "open"),
                    ToolStateEdit("set", "last_action", "use_key"),
                ),
                observed=True,
                verified=True,
                event_cost=4,
            )
            commit_state = {"door_state": "closed", "last_action": "none"}
            rollback_state = dict(commit_state)
            committed = adapter.execute(
                commit_state,
                plan=decision,
                request=request,
                observed_outcome="door_open",
            )
            rolled_back = adapter.execute(
                rollback_state,
                plan=decision,
                request=request,
                observed_outcome="alarm_triggered",
            )
            tool_commit = {
                "result": committed.to_dict(),
                "state": commit_state,
            }
            tool_rollback = {
                "result": rolled_back.to_dict(),
                "state": rollback_state,
            }
        cases[str(row["case_id"])] = {
            "decision": decision.to_dict(),
            "outcome": outcome,
            "event_memory_admission": admission,
            "action_selection": action_selection,
            "tool_commit": tool_commit,
            "tool_rollback": tool_rollback,
        }
    safe_selection = cases["safe_plan"]["action_selection"]
    safe_selection_fixture = next(
        row["action_selection"]
        for row in rows
        if str(row["case_id"]) == "safe_plan"
    )
    checks = {
        "safe_plan_accepted": cases["safe_plan"]["decision"]["accepted"] is True,
        "missing_rollback_rejected": cases["missing_rollback"]["decision"]["accepted"] is False,
        "high_risk_rejected": cases["high_risk"]["decision"]["accepted"] is False,
        "stale_goal_rejected": cases["stale_goal"]["decision"]["accepted"] is False,
        "observed_success_can_be_candidate": cases["safe_plan"]["outcome"]["event_memory_candidate_allowed"] is True,
        "observed_success_admitted": cases["safe_plan"]["event_memory_admission"]["accepted"] is True,
        "rejected_outcomes_not_admitted": all(
            cases[case_id]["event_memory_admission"] is None
            for case_id in ("missing_rollback", "high_risk", "stale_goal")
        ),
        "unexpected_outcome_rolls_back": (
            cases["unexpected_outcome"]["outcome"]["rollback_required"] is True
            and cases["unexpected_outcome"]["event_memory_admission"] is None
        ),
        "action_selection_equal_event_budget": bool(
            safe_selection
            and safe_selection["equal_event_budget"]
            and safe_selection["control"]["charged_event_budget"]
            == safe_selection["structural_feedback"]["charged_event_budget"]
            == safe_selection["event_envelope_cost"]
        ),
        "structural_feedback_improves_action_selection": bool(
            safe_selection
            and safe_selection["control"]["selected_action"]
            == safe_selection_fixture["expected_control_action"]
            and safe_selection["structural_feedback"]["selected_action"]
            == safe_selection_fixture["expected_structural_action"]
            and safe_selection["control"]["selected_action"]
            != safe_selection["structural_feedback"]["selected_action"]
        ),
        "action_selection_trace_complete": bool(
            safe_selection
            and all(
                arm["trace"].get("concept")
                and arm["trace"].get("evidence_ref")
                and arm["trace"].get("structural_prediction")
                and arm["trace"].get("expected_outcome")
                and arm["trace"].get("side_effects_executed") is False
                for arm in (
                    safe_selection["control"],
                    safe_selection["structural_feedback"],
                )
            )
        ),
        "transactional_tool_commit_verified": (
            cases["safe_plan"]["tool_commit"]["result"]["committed"] is True
            and cases["safe_plan"]["tool_commit"]["state"]["door_state"] == "open"
            and cases["safe_plan"]["tool_commit"]["result"]["trace"][
                "side_effects_executed"
            ]
            is False
        ),
        "transactional_tool_rollback_exact": (
            cases["safe_plan"]["tool_rollback"]["result"]["rolled_back"] is True
            and cases["safe_plan"]["tool_rollback"]["result"]["before_digest"]
            == cases["safe_plan"]["tool_rollback"]["result"]["restored_digest"]
            and cases["safe_plan"]["tool_rollback"]["state"]
            == {"door_state": "closed", "last_action": "none"}
        ),
        "durable_mutation_blocked": all(
            not item["decision"]["durable_mutation_allowed"] and not item["outcome"]["durable_mutation_allowed"]
            for item in cases.values()
        ),
    }
    return {
        "schema": "sara-phase25-agent-loop-benchmark-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "external_device_required": False,
        "metrics": {
            "case_count": len(cases),
            "safe_plan_acceptance": float(checks["safe_plan_accepted"]),
            "equal_budget_action_selection": float(
                checks["action_selection_equal_event_budget"]
            ),
            "structural_feedback_action_selection_gain": float(
                checks["structural_feedback_improves_action_selection"]
            ),
            "transactional_tool_boundary": float(
                checks["transactional_tool_commit_verified"]
                and checks["transactional_tool_rollback_exact"]
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
        handle.write(f"Phase 25 agent loop benchmark: {'PASS' if report['passed'] else 'FAIL'}\n")
        for key, value in report["metrics"].items():
            handle.write(f"- {key}: {value}\n")
        for key, value in report["checks"].items():
            handle.write(f"- check.{key}: {value}\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
