"""Preregistered electricity comparison preserves task and access contracts."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'data/processed/benchmark_fixtures'


def test_dependencies_and_access_are_frozen():
    raw = (FIXTURES / 'r2_electricity_learner_protocol_v1.json').read_bytes()
    assert sha256(raw).hexdigest() == '0093d72c34a72baf49f21220e6cadbef2af0993eb3b0cce366e2e495b7fef89f'
    protocol = json.loads(raw)
    assert sha256((FIXTURES / 'r2_electricity_load_feasibility_protocol_v1.json').read_bytes()).hexdigest() == protocol['feasibility_protocol_sha256']
    assert sha256((ROOT / 'workspace/evaluation/r2_electricity_load_support_v1.json').read_bytes()).hexdigest() == protocol['support_report_sha256']
    assert protocol['numeric_access_years'] == [2011, 2012, 2013]
    assert protocol['held_out_access_authorized'] is False
    assert protocol['task_definition_changes'] is False


def test_matched_routes_and_hard_caps():
    protocol = json.loads((FIXTURES / 'r2_electricity_learner_protocol_v1.json').read_text())
    assert protocol['input_contract']['same_route_trace_for_all_arms'] is True
    assert protocol['input_contract']['outcome_not_in_encoder'] is True
    assert protocol['learning']['max_routes'] == protocol['input_contract']['route_count']
    assert protocol['budget']['max_unit_ticks_per_episode'] == 95 * protocol['budget']['max_units']
    assert protocol['evaluation']['candidate_fixed_before_scoring'] == 'stateful_lif'
