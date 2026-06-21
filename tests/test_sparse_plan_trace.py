from sara_engine.reasoning.sparse_plan_trace import build_repair_materials, verify_sparse_plan_trace


def _base_case():
    return {
        "case_id": "test_plan",
        "initial_state": ["at_a", "path_a_b", "door_closed"],
        "goal": ["at_b"],
        "actions": {
            "move_a_b": {"pre": ["at_a", "path_a_b"], "add": ["at_b"], "del": ["at_a"]},
        },
        "plan": [{"action": "move_a_b"}],
        "expected_valid": True,
    }


def test_sparse_plan_trace_accepts_valid_transition():
    result = verify_sparse_plan_trace(_base_case())

    assert result.valid is True
    assert result.invalid_step_count == 0
    assert result.goal_missing == []
    assert "at_b" in result.final_state
    assert result.event_cost > 0
    assert result.state_budget_units > 0
    assert result.sparse_trace_signature


def test_sparse_plan_trace_rejects_missing_precondition():
    case = _base_case()
    case["initial_state"] = ["path_a_b"]

    result = verify_sparse_plan_trace(case)

    assert result.valid is False
    assert "step_0:precondition_unsatisfied" in result.errors
    assert result.step_results[0].preconditions_missing == ["at_a"]


def test_sparse_plan_trace_rejects_wrong_effect_and_missing_frame():
    case = _base_case()
    case["plan"] = [{"action": "move_a_b", "claimed_next_state": ["at_b"]}]

    result = verify_sparse_plan_trace(case)

    assert result.valid is False
    assert "step_0:wrong_effects" in result.errors
    assert "step_0:missing_frame_persistence" in result.errors


def test_sparse_plan_trace_builds_repair_materials_for_invalid_cases():
    case = _base_case()
    case["initial_state"] = ["path_a_b"]
    result = verify_sparse_plan_trace(case)

    repairs = build_repair_materials([case], [result])

    assert len(repairs) == 1
    assert repairs[0]["material_type"] == "plan_trace_repair"
    assert repairs[0]["observed_only"] is True
    assert repairs[0]["content"]["invalid_step_count"] == 1
