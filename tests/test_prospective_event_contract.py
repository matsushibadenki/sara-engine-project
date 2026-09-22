"""Delayed event outcomes require case-specific follow-up and cutoff-safe features."""
from datetime import datetime, timedelta, timezone

import pytest

from sara_engine.evaluation.prospective_event_contract import (
    CaseFollowup, admit_prospective_event_case,
)


T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
CUTOFF = T0 + timedelta(days=2)
FEATURES = (T0, T0 + timedelta(days=1), CUTOFF)


def admit(*, outcome=None, followup_days=12, evidence="continuous_case_observation", features=FEATURES):
    return admit_prospective_event_case(
        cutoff=CUTOFF, feature_times=features, first_outcome_at=outcome,
        followup=CaseFollowup(T0 + timedelta(days=followup_days), evidence),
        horizon=timedelta(days=7), minimum_feature_events=3)


def test_future_outcome_within_horizon_is_positive():
    result = admit(outcome=CUTOFF + timedelta(days=3), followup_days=5)
    assert result is not None and result.label == 1
    assert result.horizon_end == CUTOFF + timedelta(days=7)
    assert result.feature_event_count == 3


def test_already_outcome_and_censored_negative_abstain():
    assert admit(outcome=CUTOFF) is None
    assert admit(outcome=None, followup_days=8) is None
    with pytest.raises(ValueError, match="beyond verified"):
        admit(outcome=CUTOFF + timedelta(days=10), followup_days=8)


def test_verified_case_followup_admits_late_or_absent_outcome_as_negative():
    assert admit(outcome=None, followup_days=9).label == 0
    assert admit(outcome=CUTOFF + timedelta(days=10), followup_days=12).label == 0


def test_global_log_end_is_not_an_acceptable_followup_source():
    with pytest.raises(ValueError, match="Case-level"):
        admit(evidence="global_log_end")
    with pytest.raises(ValueError, match="Case-level"):
        admit(evidence="last_event_without_continuity")


def test_future_feature_or_misaligned_cutoff_is_rejected():
    with pytest.raises(ValueError, match="end at cutoff"):
        admit(features=FEATURES + (CUTOFF + timedelta(days=1),))
    with pytest.raises(ValueError, match="end at cutoff"):
        admit(features=(T0, CUTOFF, T0 + timedelta(days=1)))


def test_unobserved_future_outcome_is_rejected():
    with pytest.raises(ValueError, match="beyond verified"):
        admit(outcome=CUTOFF + timedelta(days=3), followup_days=4)


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        CaseFollowup(datetime(2020, 1, 8), "explicit_terminal")
    with pytest.raises(ValueError, match="timezone-aware"):
        admit_prospective_event_case(
            cutoff=datetime(2020, 1, 3), feature_times=FEATURES, first_outcome_at=None,
            followup=CaseFollowup(T0 + timedelta(days=10), "explicit_terminal"),
            horizon=timedelta(days=7), minimum_feature_events=3)
