"""Bound development coverage using only KSEA page 1 record headers."""

from calendar import monthrange
from datetime import datetime, timedelta
from typing import BinaryIO, NamedTuple


class TimestampCoverage(NamedTuple):
    indexed_minutes: int
    absent_minutes: int
    indexed_dates: int

    @property
    def status(self) -> str:
        return "excluded_by_date_bound" if self.indexed_dates < 180 else "cannot_exclude"


def _local_minute(header: bytes, month: int) -> datetime:
    if header[:9] != b"24233KSEA":
        raise ValueError("Unexpected station")
    local, utc = header[13:25], header[25:29]
    if len(utc) != 4 or not local.isdigit() or not utc.isdigit():
        raise ValueError("Invalid timestamp header")
    at = datetime.strptime(local.decode("ascii"), "%Y%m%d%H%M")
    if at.strftime("%Y%m%d%H%M").encode("ascii") != local:
        raise ValueError("Invalid local timestamp")
    if (at.year, at.month) != (2024, month):
        raise ValueError("Record outside its development month")
    expected_utc = (at + timedelta(hours=8)).strftime("%H%M").encode("ascii")
    if utc != expected_utc:
        raise ValueError("UTC and local timestamps disagree")
    return at


def bound_development_dates(months: tuple[BinaryIO, ...]) -> TimestampCoverage:
    """Return a necessary coverage bound for six January–June streams in order."""
    if len(months) != 6:
        raise ValueError("Exactly six development streams required")

    indexed = absent = dates = 0
    for month, stream in enumerate(months, 1):
        possible_minutes = monthrange(2024, month)[1] * 1440
        previous = None
        count = 0
        for _ in range(possible_minutes + 1):
            line = stream.readline(102)
            if line == b"":
                break
            # Only line framing and the station/time header have meaning here.
            record = line.removesuffix(b"\n").removesuffix(b"\r")
            if len(record) != 99 or len(line) > 101:
                raise ValueError("Invalid fixed-width record")
            at = _local_minute(record[:29], month)
            if previous is not None and at <= previous:
                raise ValueError("Timestamps must be strictly increasing")
            if previous is None or at.day != previous.day:
                dates += 1
            previous = at
            count += 1
        else:
            raise ValueError("Monthly line budget exhausted")
        indexed += count
        absent += possible_minutes - count
    return TimestampCoverage(indexed, absent, dates)
