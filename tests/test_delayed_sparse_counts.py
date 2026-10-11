"""Delayed receipt identity, causal clocks and real-change accounting."""
from dataclasses import replace
from datetime import datetime, timedelta
import pytest
from sara_engine.evaluation.delayed_sparse_counts import DelayedSparseCounts

T = datetime(2010, 11, 4)
KEYS = ('global', 'current', 'branch')


def test_overlapping_horizons_and_frozen_reads():
    m = DelayedSparseCounts()
    a = m.predict(KEYS, T, T+timedelta(seconds=60))
    b = m.predict(KEYS, T+timedelta(seconds=30), T+timedelta(seconds=90))
    assert a.probability == b.probability == .5
    m.deliver(a, a.due, lambda:1, training=True)
    c = m.predict(KEYS, a.due, T+timedelta(seconds=120))
    assert c.probability > .5 and m.changed_writes == 3
    m.deliver(b, b.due, lambda:0, training=True)
    before = m.changed_writes
    m.deliver(c, c.due, lambda:1, training=False)
    assert m.changed_writes == before == 6 and m.peak_pending == 2


@pytest.mark.parametrize('wrong', ['early', 'forged', 'capacity'])
def test_rejection_precedes_supplier(wrong):
    m = DelayedSparseCounts(capacity=2 if wrong=='capacity' else 100)
    r = m.predict(KEYS, T, T+timedelta(seconds=60))
    with pytest.raises(ValueError):
        m.deliver(replace(r) if wrong=='forged' else r, T if wrong=='early' else r.due,
                  lambda:pytest.fail('Rejected receipt accessed outcome'), training=True)
    assert not m.cells and not m.changed_writes


def test_due_order_and_prediction_boundary():
    m = DelayedSparseCounts()
    a = m.predict(KEYS, T, T+timedelta(seconds=60))
    b = m.predict(KEYS, T, T+timedelta(seconds=90))
    with pytest.raises(ValueError):m.deliver(b,b.due,lambda:1,training=True)
    with pytest.raises(ValueError):m.predict(KEYS,a.due,b.due)
    m.deliver(a,a.due,lambda:1,training=True)
    m.deliver(b,b.due,lambda:1,training=True)
    with pytest.raises(ValueError):m.deliver(b,b.due,lambda:pytest.fail('Duplicate label'),training=True)


def test_checkpoint_replay_and_integrity():
    m = DelayedSparseCounts()
    r = m.predict(KEYS,T,T+timedelta(seconds=60))
    with pytest.raises(ValueError):m.checkpoint()
    m.deliver(r,r.due,lambda:1,training=True)
    cp = m.checkpoint(); restored = DelayedSparseCounts.restore(cp)
    assert restored.predict(KEYS,r.due,r.due+timedelta(seconds=60)).probability == m.predict(KEYS,r.due,r.due+timedelta(seconds=60)).probability
    cp['changed_writes'] += 1
    with pytest.raises(ValueError):DelayedSparseCounts.restore(cp)


def test_pending_and_route_limits():
    m = DelayedSparseCounts(pending_capacity=1)
    m.predict(KEYS,T,T+timedelta(seconds=60))
    with pytest.raises(ValueError):m.predict(KEYS,T,T+timedelta(seconds=60))
    with pytest.raises(ValueError):DelayedSparseCounts().predict(('a','a','b'),T,T+timedelta(seconds=1))
