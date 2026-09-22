"""CSV extraction must gate case and time before touching descriptive cells."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256

import pytest

from sara_engine.evaluation.sealed_temporal_csv import CaseBoundary, SealedTemporalCsvReader


T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
HEADER = ("case", "time", "feature", "future_outcome")


def identity(raw_id: str) -> str:
    return sha256(f"frozen-protocol|case|{raw_id}".encode()).hexdigest()[:24]


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def reader() -> SealedTemporalCsvReader:
    return SealedTemporalCsvReader(
        header=HEADER, raw_case_column="case", event_time_column="time",
        feature_columns=("feature",), forbidden_columns=("future_outcome",),
        selected_cases={identity("train"): CaseBoundary("training", T0 + timedelta(days=2)),
                        identity("sealed"): CaseBoundary("heldout", T0 + timedelta(days=2))},
        allowed_splits=frozenset({"training"}), case_namespace="frozen-protocol",
        parse_time=parse_time)


class GuardedRow:
    def __init__(self, values: tuple[str, ...], allowed_indices: set[int]):
        self.values = values
        self.allowed_indices = allowed_indices
        self.read_indices: list[int] = []

    def __getitem__(self, index: int) -> str:
        if index not in self.allowed_indices:
            raise AssertionError(f"Column {index} was read across a seal")
        self.read_indices.append(index)
        return self.values[index]


def test_unselected_and_heldout_rows_read_only_case_identity():
    access = reader()
    for raw_id in ("unknown", "sealed"):
        row = GuardedRow((raw_id, "not parsed", "private", "outcome"), {0})
        assert access.admit(row) is None
        assert row.read_indices == [0]


def test_selected_future_row_reads_only_case_and_time():
    row = GuardedRow(("train", (T0 + timedelta(days=3)).isoformat(), "future", "outcome"), {0, 1})
    assert reader().admit(row) is None
    assert row.read_indices == [0, 1]


def test_selected_cutoff_row_reads_only_allowlisted_feature():
    row = GuardedRow(("train", (T0 + timedelta(days=2)).isoformat(), "current", "outcome"), {0, 1, 2})
    admitted = reader().admit(row)
    assert admitted.case_id == identity("train")
    assert admitted.split == "training"
    assert admitted.features == {"feature": "current"}
    assert row.read_indices == [0, 1, 2]


def test_schema_and_timezone_fail_closed():
    with pytest.raises(ValueError, match="allowlist"):
        SealedTemporalCsvReader(
            header=HEADER, raw_case_column="case", event_time_column="time",
            feature_columns=("future_outcome",), forbidden_columns=("future_outcome",),
            selected_cases={}, allowed_splits=frozenset({"training"}),
            case_namespace="frozen-protocol", parse_time=parse_time)
    with pytest.raises(ValueError, match="Duplicate CSV"):
        SealedTemporalCsvReader(
            header=("case", "case", "time", "feature"), raw_case_column="case",
            event_time_column="time", feature_columns=("feature",), forbidden_columns=(),
            selected_cases={}, allowed_splits=frozenset({"training"}),
            case_namespace="frozen-protocol", parse_time=parse_time)
    row = GuardedRow(("train", "2020-01-02T00:00:00", "feature", "outcome"), {0, 1})
    with pytest.raises(ValueError, match="timezone-aware"):
        reader().admit(row)
