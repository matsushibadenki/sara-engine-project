"""Bound development coverage using only KSEA page 1 record headers."""

from calendar import monthrange
from collections.abc import Iterator
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


def iter_development_records(
    months: tuple[BinaryIO, ...],
) -> Iterator[tuple[datetime, bytes]]:
    """Yield strictly ordered, framed records from the fixed development months."""
    if len(months) != 6:
        raise ValueError("Exactly six development streams required")

    for month, stream in enumerate(months, 1):
        possible_minutes = monthrange(2024, month)[1] * 1440
        previous = None
        for _ in range(possible_minutes + 1):
            line = stream.readline(102)
            if line == b"":
                break
            record = line.removesuffix(b"\n").removesuffix(b"\r")
            if len(record) != 99 or len(line) > 101:
                raise ValueError("Invalid fixed-width record")
            at = _local_minute(record[:29], month)
            if previous is not None and at <= previous:
                raise ValueError("Timestamps must be strictly increasing")
            previous = at
            yield at, record
        else:
            raise ValueError("Monthly line budget exhausted")


def bound_development_dates(months: tuple[BinaryIO, ...]) -> TimestampCoverage:
    """Return a necessary coverage bound for six January–June streams in order."""
    indexed = dates = 0
    previous_date = None
    for at, _ in iter_development_records(months):
        indexed += 1
        day = at.date()
        if day != previous_date:
            dates += 1
            previous_date = day
    possible_minutes = sum(monthrange(2024, month)[1] for month in range(1, 7)) * 1440
    return TimestampCoverage(indexed, possible_minutes - indexed, dates)
