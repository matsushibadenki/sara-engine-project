from sara_engine.evaluation.stage_d_contract import (
    STAGE_D_ACCEPTANCE_CANDIDATE_CHECKS,
    STAGE_D_ACCEPTANCE_CANDIDATE_METRIC_NAMES,
    STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_ACTIONS,
    STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_NEXT_STEP_HINT,
    STAGE_D_DELTA_MEMORY_PROMOTION_METRIC_NAMES,
    STAGE_D_MINIMUM_METRIC_NAMES,
    STAGE_D_REQUIRED_MINIMUM_CHECKS,
    stage_d_metric_check_name,
)


def test_stage_d_acceptance_candidates_have_stable_check_descriptions():
    assert len(STAGE_D_ACCEPTANCE_CANDIDATE_METRIC_NAMES) == len(
        set(STAGE_D_ACCEPTANCE_CANDIDATE_METRIC_NAMES)
    )
    assert len(STAGE_D_DELTA_MEMORY_PROMOTION_METRIC_NAMES) == len(
        set(STAGE_D_DELTA_MEMORY_PROMOTION_METRIC_NAMES)
    )

    for metric_name in STAGE_D_ACCEPTANCE_CANDIDATE_METRIC_NAMES:
        check_name = stage_d_metric_check_name(metric_name)
        assert check_name in STAGE_D_ACCEPTANCE_CANDIDATE_CHECKS
        assert STAGE_D_ACCEPTANCE_CANDIDATE_CHECKS[check_name].strip()


def test_stage_d_minimum_checks_have_stable_descriptions():
    assert len(STAGE_D_MINIMUM_METRIC_NAMES) == len(set(STAGE_D_MINIMUM_METRIC_NAMES))
    for metric_name in STAGE_D_MINIMUM_METRIC_NAMES:
        check_name = stage_d_metric_check_name(metric_name)
        assert check_name in STAGE_D_REQUIRED_MINIMUM_CHECKS
        assert STAGE_D_REQUIRED_MINIMUM_CHECKS[check_name].strip()


def test_stage_d_acceptance_candidate_stability_actions_are_contract_owned():
    assert STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_NEXT_STEP_HINT == (
        "review_stage_d_acceptance_candidates_for_minimum_promotion"
    )
    assert len(STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_ACTIONS) >= 3
    assert len(STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_ACTIONS) == len(
        set(STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_ACTIONS)
    )
    assert all(action.strip() for action in STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_ACTIONS)
    assert any(
        "stage_d_contract acceptance candidates" in action
        for action in STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_ACTIONS
    )
