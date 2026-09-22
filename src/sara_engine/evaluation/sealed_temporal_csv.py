"""Read CSV features only after case and decision-time admission."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class CaseBoundary:
    split: str
    cutoff: datetime


@dataclass(frozen=True)
class AdmittedCsvRow:
    case_id: str
    split: str
    event_time: datetime
    features: dict[str, str]


class SealedTemporalCsvReader:
    """Enforce ID -> membership -> event time -> cutoff -> feature access.

    The caller supplies hashed case IDs and case-specific cutoffs from a frozen
    manifest. Raw IDs and post-cutoff descriptive fields are never returned.
    """

    def __init__(
        self,
        *,
        header: Sequence[str],
        raw_case_column: str,
        event_time_column: str,
        feature_columns: Sequence[str],
        forbidden_columns: Sequence[str],
        selected_cases: Mapping[str, CaseBoundary],
        allowed_splits: frozenset[str],
        case_namespace: str,
        parse_time: Callable[[str], datetime],
        hash_length: int = 24,
    ) -> None:
        if len(header) != len(set(header)):
            raise ValueError("Duplicate CSV header is not allowed")
        if not raw_case_column or not event_time_column or raw_case_column == event_time_column:
            raise ValueError("Case and time columns must be distinct")
        if not feature_columns or len(feature_columns) != len(set(feature_columns)):
            raise ValueError("Feature columns must be nonempty and unique")
        if not allowed_splits or not case_namespace or not callable(parse_time):
            raise ValueError("Reader identity and time parser are required")
        if hash_length < 16 or hash_length > 64:
            raise ValueError("Hash length is out of range")
        if set(feature_columns).intersection(set(forbidden_columns) | {raw_case_column, event_time_column}):
            raise ValueError("Feature allowlist intersects lookup or forbidden columns")
        needed = (raw_case_column, event_time_column, *feature_columns)
        missing = set(needed).difference(header)
        if missing:
            raise ValueError("Required CSV columns are missing")
        if any(boundary.cutoff.tzinfo is None or boundary.cutoff.utcoffset() is None
               for boundary in selected_cases.values()):
            raise ValueError("Case cutoff must be timezone-aware")
        self._case_index = header.index(raw_case_column)
        self._time_index = header.index(event_time_column)
        self._feature_indices = tuple((name, header.index(name)) for name in feature_columns)
        self._selected_cases = dict(selected_cases)
        self._allowed_splits = allowed_splits
        self._namespace = case_namespace
        self._parse_time = parse_time
        self._hash_length = hash_length

    def admit(self, row: Sequence[str]) -> AdmittedCsvRow | None:
        raw_case_id = row[self._case_index]
        case_id = sha256(f"{self._namespace}|case|{raw_case_id}".encode()).hexdigest()[:self._hash_length]
        boundary = self._selected_cases.get(case_id)
        if boundary is None or boundary.split not in self._allowed_splits:
            return None
        event_time = self._parse_time(row[self._time_index])
        if event_time.tzinfo is None or event_time.utcoffset() is None:
            raise ValueError("Event time must be timezone-aware")
        if event_time > boundary.cutoff:
            return None
        features = {name: row[index] for name, index in self._feature_indices}
        return AdmittedCsvRow(case_id, boundary.split, event_time, features)
