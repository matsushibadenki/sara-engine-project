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
