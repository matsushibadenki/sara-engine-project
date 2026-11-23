from datetime import datetime, timedelta, timezone

import pytest

from sara_engine.evaluation.asos_ksea_delayed_contract import (
    DEVELOPMENT_ACQUISITION_PLAN,
    SensorMinute,
    delayed_outcomes,
)


START = datetime(2024, 1, 1)


def readings(start=START, first=80, outcome=100):
    coefficients = [first] + [90] * 11 + [outcome]
    return [
        SensorMinute(start + timedelta(minutes=i), value, "D")
        for i, value in enumerate(coefficients)
    ]


def test_fixed_acquisition_budget():
    assert len(DEVELOPMENT_ACQUISITION_PLAN) == 6
    for month, item in enumerate(DEVELOPMENT_ACQUISITION_PLAN, 1):
        assert item.url.endswith(f"/{month:02d}/asos-1min-pg1-KSEA-2024{month:02d}.dat")
    assert sum(item.max_body_bytes for item in DEVELOPMENT_ACQUISITION_PLAN) == 26_470_080


def test_same_exact_current_opposing_histories_and_threshold():
    rising, falling = list(delayed_outcomes(
        readings() + readings(START + timedelta(days=7), first=110, outcome=99)
    ))
    assert rising.current_key == falling.current_key == (90, "D", 0, 0, 0)
    assert rising.history_direction == 1 and falling.history_direction == -1
    assert rising.positive is True and falling.positive is False
    assert rising.prediction_at == START + timedelta(minutes=7)
    assert rising.due_at == START + timedelta(minutes=12)
    assert list(delayed_outcomes(readings()[:-1])) == []


@pytest.mark.parametrize("index", [0, 7, 10, 12])
def test_skip_missing_minutes_and_restart_contiguous_history(index):
    rows = readings()
    missing = rows[index]._replace(extinction=None, illumination=None)
    later = readings(START + timedelta(minutes=13))
    for replacement in ([], [missing]):
        results = list(delayed_outcomes(rows[:index] + replacement + rows[index + 1:] + later))
        assert len(results) == 13 - index
        assert results[0].prediction_at == missing.at + timedelta(minutes=8)


@pytest.mark.parametrize("at", [
    datetime(2023, 12, 31, 23, 59),
    datetime(2024, 7, 1),
    datetime(2024, 8, 1),
    START.replace(second=1),
    START.replace(tzinfo=timezone.utc),
])
def test_reject_ineligible_times(at):
    with pytest.raises(ValueError, match="development minute"):
        list(delayed_outcomes([SensorMinute(at, 1, "D")]))


@pytest.mark.parametrize("extinction,light", [(True, "D"), (-1, "D"), (1.0, "D"), (1, "M"), (None, "N")])
def test_reject_invalid_normalized_values(extinction, light):
    with pytest.raises(ValueError, match="normalized first-sensor"):
        list(delayed_outcomes([SensorMinute(START, extinction, light)]))


def test_reject_duplicate_and_regressing_times():
    row, next_row, *_ = readings()
    for rows in ([row, row], [next_row, row]):
        with pytest.raises(ValueError, match="strictly increasing"):
            list(delayed_outcomes(rows))
