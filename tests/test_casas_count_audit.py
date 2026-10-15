"""Fixed-bound support, fixed-margin attainability and read-only count audits."""
from copy import deepcopy
import gzip
import json
import pytest
from sara_engine.evaluation.casas_count_audit import classify, audit, load_snapshot, CATEGORIES
from sara_engine.evaluation.casas_evidence import evidence


def test_all_five_categories_and_strict_evidence_equivalence():
    cases = [((5,5),(5,5),'no_complement'),
             ((8,0),(8,8),'support_impossible'),
             ((100,0),(199,1),'margin_impossible'),
             ((50,50),(100,100),'observed_below_bound'),
             ((100,0),(100,100),'evidence_pass')]
    assert set(x[2] for x in cases) == set(CATEGORIES)
    for detail,current,category in cases:
        item = classify(detail,current)
        assert item['category'] == category
        assert item['evidence'] == evidence(detail,current)
    assert classify((9,0),(9,9))['category'] == 'evidence_pass'


def test_fixed_margin_maximum_matches_exhaustive_integer_allocations():
    for n in range(1,10):
        for m in range(1,10):
            for positives in range(n+m+1):
                feasible = range(max(0,positives-m),min(n,positives)+1)
                actual = max(abs(x/n-(positives-x)/m) for x in feasible)
                x = min(n,positives)
                current = (n+m-positives,positives)
                result = classify((n-x,x),current)
                assert result['margin_maximum_separation'] == actual
                assert result['evidence'] == evidence((n-x,x),current)


def state():
    current = repr(('current',(('Bedroom','ON'),2,10,0)))
    return {'arm':'pool_evidence','policy':'same_sensor_fast10_four_branches_v1',
            'counts':{'schema':1,'capacity':100000,'changed_writes':600,
                      'cells':{'global':[100,100],current:[100,100],
                               repr(('pool',current,False,False)):[100,0],
                               repr(('pool',current,False,True)):[0,100]}}}


def test_partition_conservation_opposing_histories_and_no_learning():
    source = state()
    before = deepcopy(source)
    result = audit(source)
    assert source == before
    assert result['categories']['evidence_pass'] == 2
    assert result['fit_observations_by_category']['evidence_pass'] == 200
    assert result['opposing_evidence_contexts'] == 1
    assert result['audit_learning_writes'] == 0
    assert result['existing_changed_writes'] == 600
    assert result['current_contexts'] == 1 and result['detail_branches'] == 2


def test_inconsistent_detail_partition_is_rejected_even_with_valid_total_writes():
    source = state()
    keys = list(source['counts']['cells'])
    source['counts']['cells'][keys[-1]] = [1,99]
    with pytest.raises(ValueError,match='partition'):
        audit(source)


def test_current_addresses_are_not_pooled_together():
    source = state()
    cells = source['counts']['cells']
    detail_key = next(k for k in cells if k.startswith("('pool'"))
    counts = cells.pop(detail_key)
    other_current = repr(('current',(('Bedroom','ON'),2,10,1)))
    cells[repr(('pool',other_current,False,False))] = counts
    with pytest.raises(ValueError,match='partition'):
        audit(source)


def test_invalid_counts_and_checkpoint_identity():
    for detail,current in (((0,0),(1,1)),((2,0),(1,0)),((True,1),(2,2)),((-1,1),(2,2))):
        with pytest.raises(ValueError):classify(detail,current)
    from pathlib import Path
    from sara_engine.utils.project_paths import workspace_path, ensure_parent_directory
    destination = Path(ensure_parent_directory(workspace_path('casas_count_audit_tests','state.json.gz')))
    source = state()
    source['arm'] = 'retuned'
    destination.write_bytes(gzip.compress(json.dumps(source).encode(),mtime=0))
    with pytest.raises(ValueError,match='identity'):
        load_snapshot(destination,'pool_evidence')


def test_registered_training_only_result_and_branch_mass_are_exact():
    from pathlib import Path
    from sara_engine.utils.project_paths import processed_data_path, project_path
    from sara_engine.evaluation.casas_180_source import file_sha256
    def path(suffix):
        return processed_data_path('benchmark_fixtures','casas_count_audit_'+suffix+'_v1.json')
    result = json.loads(Path(path('result')).read_text())
    reg = json.loads(Path(path('registration')).read_text())
    assert file_sha256(path('protocol')) == reg['protocol_sha256']
    for name,digest in {**reg['code'],**reg['preserved']}.items():
        assert file_sha256(project_path(name)) == digest
    assert not any(result[k] for k in ('raw_source_opened','confirmation_opened',
                                       'original_development_opened','test_opened'))
    trace_path = path('branches')+'.gz'
    assert file_sha256(trace_path) == result['branch_trace_sha256']
    branches = json.loads(gzip.decompress(Path(trace_path).read_bytes()))
    assert {a['current_counts_sha256'] for a in result['arms'].values()} == {
        '4e7a84d1aaa8b757ccf1dde3d7dc2e2366580b33d5f5badd2701e28e8346319d'}
    for arm,report in result['arms'].items():
        assert report['fit_observations'] == 17553
        assert report['existing_changed_writes'] == 52659
        assert report['audit_learning_writes'] == 0
        assert sum(report['categories'].values()) == report['detail_branches']
        assert sum(report['fit_observations_by_category'].values()) == 17553
        assert len(branches[arm]) == report['detail_branches']
        assert report['categories']['evidence_pass'] == 0
        for category in CATEGORIES:
            records = [r for r in branches[arm] if r['category'] == category]
            assert len(records) == report['categories'][category]
            assert sum(r['detail_observations'] for r in records) == report['fit_observations_by_category'][category]
    assert result['arms']['pool_evidence']['categories']['observed_below_bound'] == 112
    assert result['arms']['joint_evidence']['categories']['observed_below_bound'] == 69


def test_bounded_decompression_and_malformed_addresses_rejected(monkeypatch):
    from pathlib import Path
    from sara_engine.utils.project_paths import workspace_path, ensure_parent_directory
    from sara_engine.evaluation import casas_count_audit as module
    destination = Path(ensure_parent_directory(workspace_path('casas_count_audit_tests','bound.json.gz')))
    destination.write_bytes(gzip.compress(b' '*100,mtime=0))
    monkeypatch.setattr(module,'MAX_EXPANDED',10)
    with pytest.raises(ValueError,match='bound'):
        load_snapshot(destination,'pool_evidence')
    for key in ('not a tuple',"['detail']",'x'*4097):
        with pytest.raises(ValueError):module.parse_key(key)
