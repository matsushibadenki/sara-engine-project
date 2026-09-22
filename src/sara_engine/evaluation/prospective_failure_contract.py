"""Leakage-safe case eligibility for prospective delayed-failure experiments."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class ProspectiveFailureCase:
    observation_end: date
    outcome_window_end: date
    label: int
    feature_days: tuple[date, ...]


def admit_prospective_failure_case(
    *,
    observation_end: date,
    observed_feature_days: tuple[date, ...],
    first_failure_day: date | None,
    last_observed_day: date,
    horizon_days: int,
    minimum_feature_days: int,
) -> ProspectiveFailureCase | None:
    """Admit a case only when its outcome is observable after the feature cutoff.

    A positive requires the first failure strictly after observation_end and no
    later than the horizon. A negative requires failure-free observation through
    the entire horizon. Earlier failures and censored controls are ineligible.
    """
    if horizon_days < 1 or minimum_feature_days < 1:
        raise ValueError("Horizon and minimum feature days must be positive")
    if len(observed_feature_days) < minimum_feature_days:
        return None
    if len(set(observed_feature_days)) != len(observed_feature_days):
        raise ValueError("Feature days must be unique")
    if tuple(sorted(observed_feature_days)) != observed_feature_days:
        raise ValueError("Feature days must be chronological")
    if observed_feature_days[-1] != observation_end:
        raise ValueError("Feature window must end at observation_end")
    if last_observed_day < observation_end:
        raise ValueError("Last observation precedes the feature cutoff")
    if first_failure_day is not None and first_failure_day <= observation_end:
        return None
    outcome_end = observation_end + timedelta(days=horizon_days)
    if first_failure_day is not None and first_failure_day <= outcome_end:
        if last_observed_day < first_failure_day:
            raise ValueError("Failure day is not observable")
        label = 1
    elif last_observed_day >= outcome_end:
        label = 0
    else:
        return None
    return ProspectiveFailureCase(
        observation_end=observation_end,
        outcome_window_end=outcome_end,
        label=label,
        feature_days=observed_feature_days,
    )
