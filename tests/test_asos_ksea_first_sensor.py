from datetime import datetime, timedelta
from io import BytesIO

import pytest

from sara_engine.evaluation.asos_ksea_delayed_contract import SensorMinute, delayed_outcomes
from sara_engine.evaluation.asos_ksea_first_sensor import first_sensor_minutes

START = datetime(2024, 1, 1)


def raw(payload, at=START, station=b"24233KSEA"):
    utc = at + timedelta(hours=8)
    return station + f" SEA{at:%Y%m%d%H%M}{utc:%H%M}".encode() + payload.ljust(70) + b"\r\n"


def streams(january):
    return (BytesIO(january), *(BytesIO() for _ in range(5)))


@pytest.mark.parametrize("payload,expected", [
    (b" 0.000 N 0.999 D", (0, "N")),
    (b" 1.025 D 0.999 N \xff", (1025, "D")),
    (b" " * 70, (None, None)),
    (b" [M] M 0.999 N", (None, None)),
    (b"         0.999 N", (None, None)),
    (b" 0.999 N 227 2", (None, None)),
    (b"  0.025 D 0.999 N", (None, None)),
    (b" 0.02 D 0.999 N", (None, None)),
    (b" 0.025 M 0.999 N", (None, None)),
    (b" 0.025 D 0.999 NX", (None, None)),
])
def test_paired_prefix(payload, expected):
    assert list(first_sensor_minutes(streams(raw(payload)))) == [SensorMinute(START, *expected)]


def test_delayed_pipeline_and_unavailable_minute():
    rows = [
        raw(b" 0.100 D 0.005 N", START + timedelta(minutes=i))
        for i in range(13)
    ]
    result, = delayed_outcomes(first_sensor_minutes(streams(b"".join(rows))))
    assert result.positive
    assert result.prediction_at == START + timedelta(minutes=7)
    assert result.due_at == START + timedelta(minutes=12)

    rows[10] = raw(b" [M] M 0.999 N", START + timedelta(minutes=10))
    assert list(delayed_outcomes(first_sensor_minutes(streams(b"".join(rows))))) == []


@pytest.mark.parametrize("body,error", [
    (raw(b"", station=b"24234KSEA"), "Unexpected station"),
    (raw(b"", datetime(2024, 8, 1)), "outside its development month"),
    (raw(b"") * 2, "strictly increasing"),
])
def test_reject_invalid_headers_and_order(body, error):
    with pytest.raises(ValueError, match=error):
        list(first_sensor_minutes(streams(body)))


def test_extra_stream_rejected_before_reading():
    inputs = streams(raw(b""))
    with pytest.raises(ValueError, match="Exactly six"):
        list(first_sensor_minutes(inputs + (BytesIO(),)))
    assert inputs[0].tell() == 0
