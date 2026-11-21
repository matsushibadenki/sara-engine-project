"""Check reconstructed arithmetic against registered literal expectations."""
from fractions import Fraction
from itertools import product
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.route_arithmetic_reconstruction import reconstruct
from sara_engine.evaluation.route_arithmetic_reconstruction_v2 import reconstruct as reconstruct_v2
from sara_engine.evaluation.route_blind_arithmetic import reconstruct as blind_reconstruct

from sara_engine.evaluation.route_blind_arithmetic_v2 import reconstruct as blind_reconstruct_v2


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
@pytest.mark.parametrize("implementation", [reconstruct, blind_reconstruct, reconstruct_v2], ids=["original", "blind", "v2"])
def test_matches_registered_expectation(row, implementation):
    probability, increment = implementation(*row[:7])
    for actual, expected in zip((probability, increment), row[7:9]):
        assert type(actual) is type(expected)
        if expected is None:
            assert actual is None
        else:
            assert actual.hex() == expected.hex()


def rational_reference(tick, history, weighting, allocated, left, right, outcome):
    scale = 1 << (tick // 64)
    prior = 1 if history == "cumulative" else scale
    ratio = Fraction(right + prior, left + right + prior + prior)
    rounded_probability = float(ratio) if allocated else None
    error_probability = Fraction(rounded_probability) if allocated else Fraction(1, 2)
    target = 1 if outcome == "mass_right" else 0
    residual = float(error_probability - target)
    square = float(Fraction(residual) ** 2)
    weight = 1 if weighting == "uniform" else scale
    return rounded_probability, float(Fraction(square) * weight)


@pytest.mark.parametrize("implementation", [reconstruct_v2, blind_reconstruct_v2], ids=["v2", "blind-v2"])
def test_v2_matches_rational_reference(implementation):
    sweep = (
        (0, "cumulative", "uniform", True, left, right, side)
        for left, right, side in product(range(64), range(64), ("mass_left", "mass_right"))
    )
    cases = [tuple(row[:7]) for row in [*LITERAL_ROWS, *BOUNDARIES]]
    cases.extend(sweep)
    assert len(cases) == 8207
    for args in cases:
        for actual, expected in zip(implementation(*args), rational_reference(*args)):
            assert type(actual) is type(expected), args
            assert (actual is None if expected is None else actual.hex() == expected.hex()), args


@pytest.mark.parametrize("implementation", [reconstruct_v2, blind_reconstruct_v2], ids=["v2", "blind-v2"])
def test_v2_scale_transitions_and_count_limits(implementation):
    ticks = [0, 1023]
    for transition in range(64, 1024, 64):
        ticks.extend((transition - 1, transition))
    masses = (0, 1, 6717439, 6717440)
    allocations = [(True, left, right) for left, right in product(masses, repeat=2)]
    allocations.append((False, 0, 0))
    cases = list(product(
        ticks, ("cumulative", "discounted"), ("uniform", "public_exponential"),
        allocations, ("mass_left", "mass_right"),
    ))
    assert len(cases) == 4352
    for tick, history, weighting, (allocated, left, right), outcome in cases:
        args = (tick, history, weighting, allocated, left, right, outcome)
        for actual, expected in zip(implementation(*args), rational_reference(*args)):
            assert type(actual) is type(expected), args
            if expected is None:
                assert actual is None, args
            else:
                assert actual.hex() == expected.hex(), args
