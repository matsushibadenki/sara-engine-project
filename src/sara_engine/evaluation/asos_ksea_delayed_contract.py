"""Development-only delayed outcome extraction from normalized KSEA minutes."""

from calendar import monthrange
from collections import deque
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta
from typing import NamedTuple

_START = datetime(2024, 1, 1)
_END = datetime(2024, 7, 1)
_MINUTE = timedelta(minutes=1)
_HISTORY = 8
_LAG = 5
_THRESHOLD = 100
_ARCHIVE = "https://www.ncei.noaa.gov/data/automated-surface-observing-system-one-minute-pg1/access/2024"


class AcquisitionItem(NamedTuple):
    url: str
    max_body_bytes: int


DEVELOPMENT_ACQUISITION_PLAN = tuple(
    AcquisitionItem(
        f"{_ARCHIVE}/{month:02d}/asos-1min-pg1-KSEA-2024{month:02d}.dat",
        monthrange(2024, month)[1] * 1440 * 101,
    )
    for month in range(1, 7)
)


class SensorMinute(NamedTuple):
    at: datetime
    extinction: int | None
    illumination: str | None


class DelayedOutcome(NamedTuple):
    prediction_at: datetime
    due_at: datetime
    current_key: tuple[int, str, int, int, int]
    history_direction: int
    positive: bool


def delayed_outcomes(minutes: Iterable[SensorMinute]) -> Iterator[DelayedOutcome]:
    """Emit complete delayed examples in delivery order from first-sensor observations."""
    window: deque[tuple[int, str]] = deque(maxlen=_HISTORY + _LAG)
    previous = None
    for at, extinction, illumination in minutes:
        if at.tzinfo is not None or at.second or at.microsecond or not _START <= at < _END:
            raise ValueError("Expected a development minute in local standard time")
        if previous is not None and at <= previous:
            raise ValueError("Minutes must be strictly increasing")
        if previous is not None and at - previous != _MINUTE:
            window.clear()
        previous = at

        if extinction is None and illumination is None:
            window.clear()
            continue
        if type(extinction) is not int or extinction < 0 or illumination not in ("D", "N"):
            raise ValueError("Invalid normalized first-sensor reading")

        window.append((extinction, illumination))
        if len(window) < window.maxlen:
            continue
        prediction = at - _LAG * _MINUTE
        current, light = window[_HISTORY - 1]
        first, _ = window[0]
        yield DelayedOutcome(
            prediction, at,
            (current, light, prediction.weekday(), prediction.hour, prediction.minute // 15),
            (current > first) - (current < first),
            extinction >= _THRESHOLD,
        )
