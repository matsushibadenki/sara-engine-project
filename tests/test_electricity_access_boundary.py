"""Ensure support access cannot parse sealed electricity payloads."""
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

import pytest
from sara_engine.evaluation import electricity_access_boundary as boundary


@pytest.mark.parametrize('year', [2010, 2014, 2015, 2026])
def test_unapproved_year_never_calls_numeric_parser(monkeypatch, year):
    def forbidden(token):
        raise AssertionError('Sealed load parser was called')
    monkeypatch.setattr(boundary, 'parse_load_token', forbidden)
    timestamp, values = boundary.parse_support_row(f'"{year}-01-01 12:00:00";SECRET;malformed')
    assert timestamp.year == year
    assert values is None


def test_exact_decimal_and_invalid_observations():
    _, values = boundary.parse_support_row('"2013-01-01 12:00:00";"0,1";NaN;-1;Infinity;bad', client_count=5)
    assert values == (Decimal('0.1'), None, None, None, None)
    assert boundary.parse_load_token('0,000') == Decimal(0)


def test_allowed_year_shape_fails_closed():
    with pytest.raises(ValueError, match='client count'):
        boundary.parse_support_row('2012-01-01 12:00:00;1', client_count=2)


def test_protocol_matches_hard_access_boundary():
    root = Path(__file__).resolve().parents[1]
    raw = (root / 'data/processed/benchmark_fixtures/r2_electricity_load_feasibility_protocol_v1.json').read_bytes()
    assert sha256(raw).hexdigest() == 'a2220cf5d49ac68fd74a71f923b7c43701f190975291d4dc7fd4f2b6ed4e9157'
    protocol = json.loads(raw)
    assert frozenset(protocol['numeric_access_years']) == boundary.NUMERIC_ACCESS_YEARS
    assert protocol['access_boundary']['candidate_scoring_authorized'] is False
    assert protocol['access_boundary']['sealed_numeric_tokenization'] is False
    assert not set(protocol['sealed_years']) & boundary.NUMERIC_ACCESS_YEARS
