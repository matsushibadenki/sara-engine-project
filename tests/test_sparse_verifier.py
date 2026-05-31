from sara_engine.nn.common_spike_space import (
    build_causal_candidate_trace,
    build_event_relation_trace,
    build_reverse_reasoning_trace,
)
from sara_engine.nn.sparse_verifier import (
    SparseVerifier,
    evaluate_bounded_tree_search_trace,
    evaluate_hierarchical_reasoning_trace,
    evaluate_reasoning_forest_lane_trace,
    evaluate_self_correction_trace,
    evaluate_sparse_best_of_n_trace,
    evaluate_sparse_verifier_trace,
)


def _candidate(branch_id: str, action: str, score: float, budget: int):
    relation = build_event_relation_trace(
        cause=f"needs_gate:{action}",
        relation="projects",
        effect="release:ready",
        branch_id=branch_id,
    )
    reverse = build_reverse_reasoning_trace(
        outcome="release:blocked",
        candidate_causes=[f"{action}:missing", "needs_gate:unchanged"],
        selected_cause="needs_gate:unchanged",
        branch_id=branch_id,
    )
    return {
        "branch_id": branch_id,
        "action": action,
        "projected_state": "release:ready",
        "score": score,
        "budget": {
            "budget": budget,
            "bounded": True,
        },
        "relation_trace": relation,
        "causal_trace": build_causal_candidate_trace(
            relation_trace=relation,
            reverse_trace=reverse,
            selected_action=action,
            branch_id=branch_id,
        ),
    }


def test_sparse_verifier_ranks_grounded_low_energy_branch():
    candidates = [
        _candidate("primary", "run release gate", 0.95, 2),
        _candidate("counterfactual-1", "defer release", 0.45, 5),
    ]
    verifier = SparseVerifier(max_energy_budget=6)

    ranked = verifier.rank_candidates(
        candidates,
        evidence_texts=[
            "pytest passes before release",
            "run release gate is grounded by the current task",
        ],
    )

    assert ranked["observed_only"] is True
    assert ranked["selected_branch"] == "primary"
    assert ranked["selected_passed"] is True
    assert ranked["ranked_candidates"][0]["grounding_score"] >= 0.25
    assert ranked["ranked_candidates"][0]["trace_integrity"] == 1.0
    assert ranked["ranked_candidates"][0]["energy_score"] >= 0.5
    assert ranked["ranked_candidates"][0]["uncertainty"] <= 0.55


def test_sparse_verifier_trace_reports_observed_metrics():
    candidates = [
        _candidate("primary", "run release gate", 0.95, 2),
        _candidate("counterfactual-1", "defer release", 0.45, 5),
    ]

    trace = evaluate_sparse_verifier_trace(
        candidates,
        evidence_texts=["run release gate after pytest passes"],
        expected_branch_id="primary",
        max_energy_budget=6,
    )

    assert trace["selected_branch"] == "primary"
    assert trace["metrics"] == {
        "sparse_verifier_grounding_observed": 1.0,
        "sparse_verifier_trace_integrity_observed": 1.0,
        "sparse_verifier_energy_budget_observed": 1.0,
        "sparse_verifier_uncertainty_observed": 1.0,
        "sparse_verifier_selection_observed": 1.0,
    }


def test_sparse_best_of_n_selects_verified_branch_with_summary_alignment():
    candidates = [
        _candidate("primary", "run release gate", 0.95, 2),
        _candidate("counterfactual-1", "defer release", 0.45, 5),
        _candidate("retrieval-heavy", "run release gate with retrieved evidence", 0.60, 4),
    ]

    trace = evaluate_sparse_best_of_n_trace(
        candidates,
        evidence_texts=[
            "run release gate after pytest passes",
            "retrieved evidence supports release gate but costs more",
        ],
        expected_branch_id="primary",
        summary_text="Selected primary branch: run release gate.",
        max_n=3,
        max_energy_budget=6,
    )

    assert trace["observed_only"] is True
    assert trace["candidate_count"] == 3
    assert trace["selected_branch"] == "primary"
    assert trace["summary_matches_selection"] is True
    assert trace["metrics"] == {
        "sparse_best_of_n_bounded_count_observed": 1.0,
        "sparse_best_of_n_branch_diversity_observed": 1.0,
        "sparse_best_of_n_verifier_selection_observed": 1.0,
        "sparse_best_of_n_summary_alignment_observed": 1.0,
    }


def test_self_correction_trace_accepts_bounded_verified_repair():
    initial = _candidate("draft", "defer release", 0.25, 5)
    repair = _candidate("primary", "run release gate", 0.95, 2)
    rejected = _candidate("noisy", "ignore release evidence", 0.20, 6)

    trace = evaluate_self_correction_trace(
        initial,
        [repair, rejected],
        evidence_texts=[
            "run release gate after pytest passes",
            "release evidence supports the primary action",
        ],
        expected_branch_id="primary",
        max_loops=2,
        min_improvement=0.05,
        max_energy_budget=6,
    )

    assert trace["observed_only"] is True
    assert trace["max_loops"] == 2
    assert trace["loop_count"] == 2
    assert trace["initial_branch"] == "draft"
    assert trace["selected_branch"] == "primary"
    assert trace["correction_applied"] is True
    assert trace["loops"][0]["accepted"] is True
    assert trace["loops"][1]["accepted"] is False
    assert trace["rollback_reason"] in {"insufficient_improvement", "verifier_failed"}
    assert trace["metrics"] == {
        "self_correction_bounded_loop_observed": 1.0,
        "self_correction_improvement_observed": 1.0,
        "self_correction_rollback_reason_observed": 1.0,
        "self_correction_verifier_failure_observed": 1.0,
    }


def test_bounded_tree_search_limits_depth_branching_and_event_budget():
    candidates = [
        {**_candidate("primary", "run release gate", 0.95, 2), "depth": 1, "event_cost": 2},
        {
            **_candidate("counterfactual-1", "defer release", 0.45, 5),
            "depth": 1,
            "event_cost": 2,
        },
        {
            **_candidate("retrieval-heavy", "run release gate with retrieved evidence", 0.60, 4),
            "parent_branch_id": "primary",
            "depth": 2,
            "event_cost": 2,
        },
        {
            **_candidate("too-deep", "recursive rollout", 0.80, 3),
            "parent_branch_id": "retrieval-heavy",
            "depth": 3,
            "event_cost": 1,
        },
    ]

    trace = evaluate_bounded_tree_search_trace(
        candidates,
        evidence_texts=[
            "run release gate after pytest passes",
            "retrieved evidence supports release gate but costs more",
        ],
        expected_branch_id="primary",
        max_depth=2,
        max_branch_factor=2,
        max_event_budget=6,
        max_energy_budget=6,
    )

    assert trace["observed_only"] is True
    assert trace["selected_branch"] == "primary"
    assert trace["candidate_count"] == 3
    assert trace["dropped_count"] == 1
    assert trace["dropped_candidates"][0]["drop_reason"] == "depth_limit"
    assert trace["event_budget_used"] == 6
    assert trace["metrics"] == {
        "bounded_tree_search_depth_observed": 1.0,
        "bounded_tree_search_branch_factor_observed": 1.0,
        "bounded_tree_search_event_budget_observed": 1.0,
        "bounded_tree_search_verifier_selection_observed": 1.0,
    }


def test_reasoning_forest_lane_uses_read_only_snapshots_and_reasoned_selection():
    lanes = [
        {
            **_candidate("primary", "run release gate", 0.95, 2),
            "lane_id": "memory-prior",
            "snapshot": {"read_only": True, "mutation_count": 0},
            "selection_reason": "primary branch has grounded release gate evidence",
        },
        {
            **_candidate("counterfactual-1", "defer release", 0.45, 5),
            "lane_id": "counterfactual",
            "snapshot": {"read_only": True, "mutation_count": 0},
            "selection_reason": "counterfactual branch preserves risk review",
        },
        {
            **_candidate("retrieval-heavy", "run release gate with retrieved evidence", 0.60, 4),
            "lane_id": "retrieval",
            "snapshot": {"read_only": True, "mutation_count": 0},
            "selection_reason": "retrieval branch has extra evidence with higher cost",
        },
    ]

    trace = evaluate_reasoning_forest_lane_trace(
        lanes,
        evidence_texts=[
            "run release gate after pytest passes",
            "retrieved evidence supports release gate but costs more",
        ],
        expected_branch_id="primary",
        max_lanes=3,
        max_energy_budget=6,
    )

    assert trace["observed_only"] is True
    assert trace["lane_count"] == 3
    assert trace["selected_branch"] == "primary"
    assert all(item["snapshot_read_only"] for item in trace["lane_summaries"])
    assert trace["metrics"] == {
        "reasoning_forest_lane_bounded_count_observed": 1.0,
        "reasoning_forest_lane_read_only_snapshot_observed": 1.0,
        "reasoning_forest_lane_diversity_observed": 1.0,
        "reasoning_forest_lane_verifier_selection_observed": 1.0,
        "reasoning_forest_lane_selection_reason_observed": 1.0,
    }


def test_hierarchical_reasoning_trace_layers_instruction_execution_and_verification():
    candidates = [
        _candidate("primary", "run release gate", 0.95, 2),
        _candidate("counterfactual-1", "defer release", 0.45, 5),
    ]

    trace = evaluate_hierarchical_reasoning_trace(
        {
            "event_type": "instruction_event",
            "instruction_id": "inst-release-gate",
            "instruction": "run release gate",
            "target_branch_id": "primary",
        },
        candidates,
        evidence_texts=["run release gate after pytest passes"],
        expected_branch_id="primary",
        max_execution_steps=3,
        max_energy_budget=6,
    )

    assert trace["observed_only"] is True
    assert trace["selected_branch"] == "primary"
    assert trace["plan_execution_alignment"] is True
    assert trace["verification_trace"]["selected_passed"] is True
    assert trace["metrics"] == {
        "hierarchical_reasoning_instruction_observed": 1.0,
        "hierarchical_reasoning_execution_trace_observed": 1.0,
        "hierarchical_reasoning_verification_trace_observed": 1.0,
        "hierarchical_reasoning_plan_alignment_observed": 1.0,
    }
