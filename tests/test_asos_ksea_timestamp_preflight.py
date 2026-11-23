from datetime import datetime, timedelta
from io import BytesIO

import pytest

from sara_engine.evaluation.asos_ksea_timestamp_preflight import bound_development_dates


def record(at):
    utc = at + timedelta(hours=8)
    return f"24233KSEA SEA{at:%Y%m%d%H%M}{utc:%H%M}".encode() + b"\xff" * 70 + b"\r\n"


def streams(january=b"", february=b"", june=b""):
    return tuple(BytesIO(body) for body in (january, february, b"", b"", b"", june))


def test_calendar_limits_leap_day_and_opaque_fields():
    result = bound_development_dates(streams(
        record(datetime(2024, 1, 1)) + record(datetime(2024, 1, 31, 23, 59)),
        record(datetime(2024, 2, 1)) + record(datetime(2024, 2, 29)),
        record(datetime(2024, 6, 30, 23, 59)),
    ))
    assert result == (5, 262075, 5)
    assert result.status == "excluded_by_date_bound"


@pytest.mark.parametrize("days,status", [(179, "excluded_by_date_bound"), (180, "cannot_exclude")])
def test_bound_threshold(days, status):
    bodies = [bytearray() for _ in range(6)]
    for offset in range(days):
        at = datetime(2024, 1, 1) + timedelta(days=offset)
        bodies[at.month - 1].extend(record(at))
    result = bound_development_dates(tuple(BytesIO(body) for body in bodies))
    assert result.indexed_dates == days
    assert result.status == status


JAN = record(datetime(2024, 1, 1))


@pytest.mark.parametrize("body", [
    JAN.replace(b"KSEA", b"KPDX"),
    JAN.replace(b"24233", b"24234"),
    JAN.replace(b"20240101", b"20240132"),
    JAN.replace(b"00000800", b"24000800"),
    JAN.replace(b"0800", b"0700"),
    JAN + JAN,
    record(datetime(2024, 1, 2)) + JAN,
    record(datetime(2024, 2, 1)),
    b"x" * 150 + b"\n",
    JAN[:29] + b"\n",
])
def test_rejects_malformed_or_out_of_order_rows(body):
    with pytest.raises(ValueError):
        bound_development_dates(streams(body))


def test_excluded_month_cannot_be_read_as_development():
    with pytest.raises(ValueError, match="outside its development month"):
        bound_development_dates(streams(june=record(datetime(2024, 7, 1))))


def test_extra_stream_is_rejected_before_reading():
    source = streams(JAN)
    with pytest.raises(ValueError, match="Exactly six"):
        bound_development_dates(source + (BytesIO(),))
    assert source[0].tell() == 0
