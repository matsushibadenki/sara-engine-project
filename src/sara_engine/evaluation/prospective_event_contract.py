"""Case-level observation contract for prospective event outcomes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


FOLLOWUP_EVIDENCE = ("explicit_terminal", "continuous_case_observation")


@dataclass(frozen=True)
class CaseFollowup:
    """Verified observation for this case, never a dataset-wide maximum time."""

    observed_through: datetime
    evidence: str

    def __post_init__(self) -> None:
        if self.observed_through.tzinfo is None or self.observed_through.utcoffset() is None:
            raise ValueError("Case follow-up must be timezone-aware")
        if self.evidence not in FOLLOWUP_EVIDENCE:
            raise ValueError("Case-level follow-up evidence is required")


@dataclass(frozen=True)
class ProspectiveEventCase:
    cutoff: datetime
    horizon_end: datetime
    label: int
    feature_event_count: int


def admit_prospective_event_case(
    *,
    cutoff: datetime,
    feature_times: tuple[datetime, ...],
    first_outcome_at: datetime | None,
    followup: CaseFollowup,
    horizon: timedelta,
    minimum_feature_events: int,
) -> ProspectiveEventCase | None:
    """Admit an at-risk case only when its future label is individually observable.

    A positive needs a post-cutoff outcome inside the horizon. A negative needs
    verified case-level observation through the whole horizon. Already-outcome
    and censored cases abstain. Global dataset end is never accepted as evidence.
    """
    if horizon <= timedelta(0) or minimum_feature_events < 1:
        raise ValueError("Horizon and feature minimum must be positive")
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("Cutoff must be timezone-aware")
    if len(feature_times) < minimum_feature_events:
        return None
    if any(time.tzinfo is None or time.utcoffset() is None for time in feature_times):
        raise ValueError("Feature timestamps must be timezone-aware")
    if tuple(sorted(feature_times)) != feature_times or feature_times[-1] != cutoff:
        raise ValueError("Feature history must be chronological and end at cutoff")
    if followup.observed_through < cutoff:
        raise ValueError("Case follow-up precedes cutoff")
    if first_outcome_at is not None:
        if first_outcome_at.tzinfo is None or first_outcome_at.utcoffset() is None:
            raise ValueError("Outcome timestamp must be timezone-aware")
        if first_outcome_at > followup.observed_through:
            raise ValueError("Outcome is beyond verified case follow-up")
        if first_outcome_at <= cutoff:
            return None
    horizon_end = cutoff + horizon
    if first_outcome_at is not None and first_outcome_at <= horizon_end:
        label = 1
    elif followup.observed_through >= horizon_end:
        label = 0
    else:
        return None
    return ProspectiveEventCase(cutoff, horizon_end, label, len(feature_times))
