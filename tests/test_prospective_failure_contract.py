"""Prospective failure cases cannot use terminal-day rows or censored controls."""
from datetime import date, timedelta

import pytest

from sara_engine.evaluation.prospective_failure_contract import (
    admit_prospective_failure_case,
)


CUTOFF = date(2014, 9, 1)
DAYS = tuple(CUTOFF - timedelta(days=offset) for offset in reversed(range(30)))


def _case(*, failure=None, last=None, days=DAYS):
    return admit_prospective_failure_case(
        observation_end=CUTOFF,
        observed_feature_days=days,
        first_failure_day=failure,
        last_observed_day=last or CUTOFF + timedelta(days=8),
        horizon_days=7,
        minimum_feature_days=30,
    )


def test_future_failure_is_positive_without_terminal_day_feature():
    result = _case(failure=CUTOFF + timedelta(days=3))
    assert result is not None
    assert result.label == 1
    assert result.feature_days[-1] < CUTOFF + timedelta(days=3)


def test_failure_on_cutoff_is_not_prospective():
    assert _case(failure=CUTOFF) is None


def test_censored_negative_is_not_assumed_healthy():
    assert _case(last=CUTOFF + timedelta(days=4)) is None


def test_fully_observed_failure_free_control_is_negative():
    result = _case()
    assert result is not None and result.label == 0


def test_later_failure_is_negative_only_when_full_horizon_observed():
    result = _case(failure=CUTOFF + timedelta(days=10),
                   last=CUTOFF + timedelta(days=8))
    assert result is not None and result.label == 0


def test_incomplete_or_misaligned_feature_window_is_rejected():
    assert _case(days=DAYS[:-1]) is None
    with pytest.raises(ValueError, match="end at observation_end"):
        _case(days=tuple(day - timedelta(days=1) for day in DAYS))
