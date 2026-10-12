"""Receipt-local admission equivalence without retained-state iteration."""
from datetime import datetime,timedelta
import pytest
from sara_engine.evaluation.delayed_sparse_counts import DelayedSparseCounts
from sara_engine.evaluation.sparse_count_replay import SparseCapacityCounts

T=datetime(2011,6,15)
KEYS=('global','current','detail')


def test_same_changed_counts_and_checkpoint():
    reference=DelayedSparseCounts();candidate=SparseCapacityCounts()
    for i in range(12):
        cutoff=T+timedelta(seconds=60*i);due=cutoff+timedelta(seconds=30)
        a=reference.predict(KEYS,cutoff,due);b=candidate.predict(KEYS,cutoff,due)
        assert a.probability==b.probability
        for m,r in ((reference,a),(candidate,b)):m.deliver(r,due,lambda:i%2,training=True)
    assert reference.checkpoint()==candidate.checkpoint()


@pytest.mark.parametrize('fault',['early','capacity','invalid_label'])
def test_same_rejection_and_no_partial_changes(fault):
    for cls in (DelayedSparseCounts,SparseCapacityCounts):
        model=cls(capacity=2 if fault=='capacity' else 100)
        r=model.predict(KEYS,T,T+timedelta(seconds=30))
        with pytest.raises(ValueError):
            model.deliver(r,T if fault=='early' else r.due,
                          lambda:7 if fault=='invalid_label' else pytest.fail('Rejected receipt read outcome'),training=True)
        assert not model.cells and model.changed_writes==0


def test_only_three_capacity_probes_and_no_global_iteration():
    class NoScan(dict):
        probes=0
        def keys(self):raise AssertionError('Global state scan')
        def __iter__(self):raise AssertionError('Global state scan')
        def __contains__(self,key):
            self.probes+=1
            return super().__contains__(key)
    model=SparseCapacityCounts(capacity=10003)
    model.cells=NoScan({str(i):[0,0] for i in range(10000)})
    r=model.predict(KEYS,T,T+timedelta(seconds=30));model.deliver(r,r.due,lambda:1,training=True)
    assert model.cells.probes==3 and model.changed_writes==3
    r=model.predict(KEYS,r.due,r.due+timedelta(seconds=30));model.deliver(r,r.due,lambda:1,training=False)
    assert model.cells.probes==3 and model.changed_writes==3

def test_full_frozen_experiment_equivalence_receipt():
    import json
    from pathlib import Path
    from sara_engine.utils.project_paths import processed_data_path,project_path
    from sara_engine.evaluation.casas_180_source import file_sha256
    reg=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_sparse_replay_registration_v1.json')).read_text())
    for p,h in reg['code'].items():assert file_sha256(project_path(p))==h
    report=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_sparse_replay_result_v1.json')).read_text())
    assert report['all_gates_metrics_traces_equal'] and report['all_eight_checkpoints_equal']
    assert report['changed_writes_per_arm']==56028 and not report['test_opened']
    assert report['original_result_sha256']==file_sha256(processed_data_path('benchmark_fixtures','casas_180_result_v1.json'))
