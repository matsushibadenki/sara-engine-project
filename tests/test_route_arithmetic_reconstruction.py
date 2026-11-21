"""Check reconstructed arithmetic against registered literal expectations."""
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.route_arithmetic_reconstruction import reconstruct


FIXTURE = (
    Path(__file__).resolve().parents[1] / "data/processed/benchmark_fixtures"
    / "route_error_binding_preflight_selection_v1.json"
)
LITERAL_ROWS = json.loads(FIXTURE.read_text(encoding="utf-8"))["local_rows"]
BOUNDARIES = [
    (63, "discounted", "public_exponential", True, 2, 0, "mass_left", 0.25, 0.0625),
    (64, "discounted", "public_exponential", True, 4, 0, "mass_right", 0.25, 1.125),
    (1023, "cumulative", "public_exponential", False, 0, 0, "mass_left", None, 8192.0),
    (1023, "cumulative", "public_exponential", True, 0, 0, "mass_left", 0.5, 8192.0),
]


@pytest.mark.parametrize("row", [*LITERAL_ROWS, *BOUNDARIES])
def test_matches_registered_expectation(row):
    probability, increment = reconstruct(*row[:7])
    for actual, expected in zip((probability, increment), row[7:9]):
        assert type(actual) is type(expected)
        if expected is None:
            assert actual is None
        else:
            assert actual.hex() == expected.hex()
