"""Saved-report-only regression checks; no learner dependencies or execution."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys

import pytest

from sara_engine.evaluation import reactivation_codec as codec
from scripts.eval import reactivation_codec as runner


@pytest.fixture(scope='module')
def saved():
    raw = Path(runner.path('selection').parent/runner.INPUTS[0]).read_bytes()
    return codec.loads(raw)


def test_saved_roundtrip_and_only_known_tuple_reconstruction(saved):
    c = codec.ReportCodec(saved)
    before = deepcopy(saved)
    reconstructed, paths = codec.reconstruct_pools(saved)
    assert len(paths) == 80
    assert saved != reconstructed
    assert c.equivalent(saved, reconstructed)
    assert c.equivalent(saved, codec.loads(c.encode(saved)))
    assert saved == before


def test_all_finite_registered_counterexamples_pass_without_numeric_replay_claim():
    result = runner.evaluate()
    assert list(sorted(result['checks'])) == sorted(runner.read('selection')['finite_checks'])
    assert len(result['checks']) == 19 and all(result['checks'].values())
    assert result['codec_preflight_passed']
    assert not result['full_replay_verified']
    assert result['numeric_trace_equality'] is None
    assert not result['original_component_passed']
    assert result['learner_executions'] == result['new_predictions_scored'] == result['new_measurements'] == 0
    assert not result['study_restart_authorized']


@pytest.mark.parametrize('value', [0, 1, 0.0, 1.0, None, 'false', []])
def test_boolean_type_never_coerces(saved, value):
    c = codec.ReportCodec(saved)
    variant = deepcopy(saved)
    variant['component_passed'] = value
    with pytest.raises(ValueError): c.encode(variant)


def test_changed_boolean_is_detected_as_difference(saved):
    variant = deepcopy(saved)
    variant['component_passed'] = True
    assert not codec.ReportCodec(saved).equivalent(saved, variant)


@pytest.mark.parametrize('path', ['cases', 'blocks', 'final_pool', 'null_pool'])
def test_tuple_exception_never_leaks_to_other_paths(saved, path):
    c = codec.ReportCodec(saved)
    variant = deepcopy(saved)
    arm = variant['cases'][0]['arms']['permanent']
    if path == 'cases': variant['cases'] = tuple(variant['cases'])
    elif path == 'blocks': arm['blocks'] = tuple(arm['blocks'])
    elif path == 'final_pool': arm['final_pool'] = tuple(arm['final_pool'])
    else: variant['cases'][0]['arms']['round_robin']['initial_pool'] = ()
    with pytest.raises(ValueError): c.encode(variant)


@pytest.mark.parametrize('change', ['missing_case', 'extra_field', 'missing_field', 'pool_item', 'pool_order', 'hash', 'float'])
def test_report_mutation_is_never_hidden(saved, change):
    variant = deepcopy(saved)
    arm = variant['cases'][0]['arms']['permanent']
    if change == 'missing_case': variant['cases'].pop()
    elif change == 'extra_field': arm['new'] = 0
    elif change == 'missing_field': del arm['state_sha256']
    elif change == 'pool_item': arm['initial_pool'][0] += 1
    elif change == 'pool_order': arm['initial_pool'].reverse()
    elif change == 'hash': arm['state_sha256'] = '0'*64
    else: arm['tail_brier'] += 1e-12
    try:
        equal = codec.ReportCodec(saved).equivalent(saved, variant)
    except ValueError:
        equal = False
    assert not equal


@pytest.mark.parametrize('raw', [b'{"a":0,"a":1}', b'{"a":{"b":0,"b":1}}', b'NaN', b'Infinity', b'-Infinity', b'1e999', b'9223372036854775808', b'"\xff"'])
def test_decoder_rejects_lossy_or_invalid_json(raw):
    with pytest.raises(ValueError): codec.loads(raw)


def test_bounds_and_signed_zero_are_explicit(saved):
    raw = codec.ReportCodec(saved).encode(saved)
    for kwargs in ({'max_bytes': 16}, {'max_nodes': 10}, {'max_depth': 2}, {'max_bytes': True}, {'max_nodes': 100001}):
        with pytest.raises(ValueError): codec.loads(raw, **kwargs)
    positive, negative = deepcopy(saved), deepcopy(saved)
    positive['cases'][0]['arms']['permanent']['tail_brier'] = 0.0
    negative['cases'][0]['arms']['permanent']['tail_brier'] = -0.0
    assert not codec.ReportCodec(saved).equivalent(positive, negative)


def test_key_order_is_ignored_but_input_and_template_are_not_mutated(saved):
    template = deepcopy(saved)
    c = codec.ReportCodec(template)
    template['cases'] = []
    assert c.equivalent(saved, dict(reversed(list(saved.items()))))
    assert len(saved['cases']) == 20


def test_no_learner_execution_or_import_path():
    allowed = {'copy', 'hashlib', 'json', 'math'}
    tree = ast.parse(inspect.getsource(codec))
    names = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    names.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert names <= allowed
    assert 'sara_engine.evaluation.sparse_route_reactivation' not in sys.modules
    source = inspect.getsource(runner)
    assert 'sparse_route_reactivation import' not in source
    assert 'subprocess' not in source


def test_existing_failure_and_one_shot_preflight_are_preserved():
    folder = runner.path('selection').parent
    assert not (folder/'sparse_route_reactivation_replay_v1.json').exists()
    failure = codec.loads((folder/runner.INPUTS[1]).read_bytes())
    assert not failure['full_replay_verified'] and failure['numeric_trace_equality'] is None
    if runner.path('registration').exists(): runner.verify()
    if runner.path('result').exists():
        assert runner.read('result') == runner.evaluate()
        with pytest.raises(FileExistsError): runner.main('execute')
