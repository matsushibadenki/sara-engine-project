"""Backend-independent, input-only temporal statistics for research readouts."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TemporalInformation:
    event_count: int
    # Each record is (route, first ordinal, last ordinal, count).
    records: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self):
        if type(self.event_count) is not int or not 1 <= self.event_count <= 95:
            raise ValueError('Event count outside contract')
        if type(self.records) is not tuple or not 1 <= len(self.records) <= 24:
            raise ValueError('Record budget outside contract')
        previous = -1
        for record in self.records:
            if type(record) is not tuple or len(record) != 4 or any(type(value) is not int for value in record):
                raise ValueError('Invalid integer record')
            route, first, last, count = record
            if not previous < route < 24 or not 0 <= first <= last < self.event_count or not 1 <= count <= last - first + 1:
                raise ValueError('Invalid temporal record')
            previous = route
        if sum(record[3] for record in self.records) != self.event_count:
            raise ValueError('Counts do not cover events')
        if len({record[1] for record in self.records}) != len(self.records) or len({record[2] for record in self.records}) != len(self.records):
            raise ValueError('Conflicting route ordinals')

    def wire_bytes(self):
        """At most 101 bytes; this is a diagnostic format, not authenticated state."""
        return b'TI1' + bytes((self.event_count, len(self.records))) + bytes(
            value for record in self.records for value in record)

    def readout_routes(self):
        """Fixed vocabulary of 72 channels; no labels or future observations."""
        return tuple(item for route, first, last, count in self.records for item in (
            (3 * route, 1.0 / (1 + first)),
            (3 * route + 1, (1 + last) / 95.0),
            (3 * route + 2, count / 95.0)))


class TemporalInformationObserver:
    """At most 24 records; updates only the arriving input route."""
    def __init__(self):
        self._records = {}
        self.event_count = 0

    def observe(self, route):
        # Reject before mutation; callers need not materialize event histories.
        if type(route) is not int or not 0 <= route < 24:
            raise ValueError('Route outside contract')
        if self.event_count >= 95:
            raise ValueError('Event capacity exceeded')
        if route in self._records:
            first, _, count = self._records[route]
            self._records[route] = (first, self.event_count, count + 1)
        else:
            self._records[route] = (self.event_count, self.event_count, 1)
        self.event_count += 1

    def snapshot(self):
        return TemporalInformation(self.event_count, tuple(
            (route, *values) for route, values in sorted(self._records.items())))
