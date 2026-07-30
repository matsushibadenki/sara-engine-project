from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "phase25_agent_loop_benchmark.py"
    spec = importlib.util.spec_from_file_location("phase25_agent_loop_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase25_agent_loop_benchmark_passes():
    module = _load_module()
    fixture = PROJECT_ROOT / "data" / "processed" / "benchmark_fixtures" / "phase25_agent_cases.jsonl"
    report = module.build_report(module._load(str(fixture)))

    assert report["passed"] is True
    assert report["checks"]["durable_mutation_blocked"] is True
    assert report["checks"]["observed_success_admitted"] is True
    assert report["checks"]["rejected_outcomes_not_admitted"] is True
    assert report["checks"]["unexpected_outcome_rolls_back"] is True
    assert report["checks"]["action_selection_equal_event_budget"] is True
    assert report["checks"]["structural_feedback_improves_action_selection"] is True
    assert report["checks"]["action_selection_trace_complete"] is True
    assert report["checks"]["transactional_tool_commit_verified"] is True
    assert report["checks"]["transactional_tool_rollback_exact"] is True
    assert report["checks"]["candidate_execution_resumable"] is True
    assert report["checks"]["judging_fork_read_only"] is True
    assert report["checks"]["stale_candidate_resume_blocked"] is True
    assert report["checks"]["typed_tool_pairing_commits_exact_batch"] is True
    assert report["checks"]["reordered_tool_pairing_blocks_commit"] is True
    assert report["checks"]["partial_rollout_round_robin"] is True
    assert report["checks"]["partial_rollout_staleness_bounded"] is True
    assert report["checks"]["stale_partial_rollout_resume_blocked"] is True


def test_phase25_agent_loop_rejects_goal_change():
    from sara_engine.agent.bounded_agent_loop import BoundedAgentLoop

    decision = BoundedAgentLoop().evaluate_plan(
        goal="new_goal",
        structural_prediction="ready",
        expected_outcome="done",
        rollback_action="undo",
        risk=0.1,
        active_goal="old_goal",
        plan_case={
            "initial_state": ["ready"],
            "goal": ["done"],
            "actions": {"finish": {"pre": ["ready"], "add": ["done"], "del": []}},
            "plan": [{"action": "finish"}],
        },
    )

    assert decision.accepted is False
    assert "stale_goal" in decision.trace["errors"]


def test_phase25_only_verified_observed_outcomes_create_candidates():
    from sara_engine.agent.bounded_agent_loop import BoundedAgentLoop

    loop = BoundedAgentLoop()
    decision = loop.evaluate_plan(
        goal="goal",
        structural_prediction="ready",
        expected_outcome="done",
        rollback_action="undo",
        risk=0.1,
        active_goal="goal",
        plan_case={
            "initial_state": ["ready"],
            "goal": ["done"],
            "actions": {"finish": {"pre": ["ready"], "add": ["done"], "del": []}},
            "plan": [{"action": "finish"}],
        },
    )

    assert loop.outcome_event_state_candidate(decision, observed_outcome="wrong", source_ref="fixture") is None
    assert loop.outcome_event_state_candidate(
        decision,
        observed_outcome="done",
        source_ref="fixture",
    ) is None
    candidate = loop.outcome_event_state_candidate(
        decision,
        observed_outcome="done",
        source_ref="fixture",
        observation_verified=True,
        observation_evidence={"sensor": "fixture", "outcome": "done"},
    )
    assert candidate is not None
    assert candidate.verification_receipt is not None


def _action_candidates():
    return (
        {
            "action": "force_door",
            "base_score": 0.8,
            "risk": 0.4,
            "concept": "door_access",
            "evidence_ref": "fixture:force",
            "structural_prediction": "door_may_open",
            "expected_outcome": "door_open",
            "event_cost": 1,
        },
        {
            "action": "use_key",
            "base_score": 0.7,
            "risk": 0.2,
            "concept": "door_access",
            "evidence_ref": "fixture:key",
            "structural_prediction": "key_matches_lock",
            "expected_outcome": "door_open",
            "event_cost": 1,
        },
    )


def _structural_feedback():
    return (
        {
            "action": "force_door",
            "verified": True,
            "feedback_stable": True,
            "contradicted": True,
            "confidence": 0.9,
            "source_ref": "fixture:constraint",
            "event_cost": 1,
        },
        {
            "action": "use_key",
            "verified": True,
            "feedback_stable": True,
            "contradicted": False,
            "confidence": 0.95,
            "source_ref": "fixture:match",
            "event_cost": 1,
        },
    )


def test_action_selection_ablation_uses_one_equal_cost_event_envelope():
    from sara_engine.agent.bounded_agent_loop import BoundedAgentLoop

    result = BoundedAgentLoop().compare_action_selection(
        candidates=_action_candidates(),
        structural_feedback=_structural_feedback(),
        event_budget_per_arm=4,
    )

    assert result.abstained is False
    assert result.equal_event_budget is True
    assert result.control.selected_action == "force_door"
    assert result.structural_feedback.selected_action == "use_key"
    assert result.control.charged_event_budget == 4
    assert result.structural_feedback.charged_event_budget == 4
    assert result.structural_feedback.trace["feedback_refs"] == ["fixture:match"]
    assert result.durable_mutation_allowed is False


def test_action_selection_ignores_unverified_or_unstable_feedback_content():
    from sara_engine.agent.bounded_agent_loop import BoundedAgentLoop

    feedback = tuple(
        {**item, "verified": False, "feedback_stable": False}
        for item in _structural_feedback()
    )
    result = BoundedAgentLoop().compare_action_selection(
        candidates=_action_candidates(),
        structural_feedback=feedback,
        event_budget_per_arm=4,
    )

    assert result.control.selected_action == "force_door"
    assert result.structural_feedback.selected_action == "force_door"
    assert result.structural_feedback.trace["feedback_refs"] == []


def test_action_selection_abstains_on_malformed_and_budget_exceeded_inputs():
    from sara_engine.agent.bounded_agent_loop import BoundedAgentLoop

    malformed = BoundedAgentLoop().compare_action_selection(
        candidates=({"action": "missing_trace", "event_cost": 1},),
        structural_feedback=(),
        event_budget_per_arm=1,
    )
    over_event_budget = BoundedAgentLoop().compare_action_selection(
        candidates=_action_candidates(),
        structural_feedback=_structural_feedback(),
        event_budget_per_arm=3,
    )
    over_state_budget = BoundedAgentLoop().compare_action_selection(
        candidates=_action_candidates(),
        structural_feedback=_structural_feedback(),
        event_budget_per_arm=4,
        max_state_budget_units=3,
    )

    assert malformed.abstained is True
    assert malformed.reason == "malformed_action_candidate"
    assert over_event_budget.reason == "action_selection_event_budget_exceeded"
    assert over_state_budget.reason == "action_selection_state_budget_exceeded"
