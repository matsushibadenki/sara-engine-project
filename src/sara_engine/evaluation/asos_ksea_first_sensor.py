"""Normalize the documented paired visibility layout from KSEA page 1."""

import re
from collections.abc import Iterator
from typing import BinaryIO

from sara_engine.evaluation.asos_ksea_delayed_contract import SensorMinute
from sara_engine.evaluation.asos_ksea_timestamp_preflight import iter_development_records

_PAIRED_VISIBILITY = re.compile(rb" ([0-9])\.([0-9]{3}) ([DN]) [0-9]\.[0-9]{3} [DN](?= |$)")


def first_sensor_minutes(months: tuple[BinaryIO, ...]) -> Iterator[SensorMinute]:
    """Yield exact thousandths or an unavailable minute for an unsupported layout."""
    for at, record in iter_development_records(months):
        fields = _PAIRED_VISIBILITY.match(record, 29)
        if fields is None:
            yield SensorMinute(at, None, None)
            continue

        whole, fraction, illumination = fields.groups()
        yield SensorMinute(at, 1000 * int(whole) + int(fraction), illumination.decode("ascii"))
