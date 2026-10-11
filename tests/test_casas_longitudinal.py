"""Source sealing, annotation eligibility and update-matched delayed comparison."""
from dataclasses import replace
from datetime import datetime, timedelta
from sara_engine.evaluation.casas_longitudinal import (
    Example, addresses, extract_lines, phase, support,
)
from sara_engine.evaluation.delayed_sparse_counts import DelayedSparseCounts


def lines():
    t = datetime(2012,7,22,10)
    result=[]
    for i in range(25):
        stamp=t+timedelta(seconds=60*i)
        annotation='Sleep="begin"' if i==0 else 'Sleep="end"' if i==14 else 'Cook="begin"' if i==15 else ''
        result.append(stamp.strftime('%Y-%m-%d,%H:%M:%S')+',Bedroom,ON,'+annotation+'\n')
    result.append('2012-09-04,00:00:00,SEALED,INVALID,NotEvenAnAnnotation\n')
    return result


def test_sealed_annotation_and_missing_are_not_negative():
    examples,audit=extract_lines(lines())
    assert audit['sealed_timestamp_only_rows']==1
    assert {e.label for e in examples['train']}=={0,1}
    assert audit['unannotated_outcomes']>0
    assert all(300<=(e.due-e.cutoff).total_seconds()<=600 for e in examples['train'])


def test_support_requires_both_classes_on_many_days():
    examples,_=extract_lines(lines())
    assert not support(examples)['passed']


def test_joint_retains_current_and_gap_destroyed_controls():
    t=datetime(2012,7,22,10)
    h=(('Bedroom','ON'),)*8
    e=Example(t,t+timedelta(seconds=300),h,(0,)*7,1)
    other=replace(e,gaps=(3,)*7)
    assert addresses(e,'current')==addresses(other,'current')
    assert addresses(e,'histogram')==addresses(other,'histogram')
    assert addresses(e,'transition')==addresses(other,'transition')
    assert addresses(e,'joint')!=addresses(other,'joint')
    assert addresses(e,'constant_gap')==addresses(other,'constant_gap')
    assert addresses(e,'joint')==addresses(e,'scalar_joint')


def test_frozen_delayed_receipt_roundtrip():
    examples,_=extract_lines(lines())
    rows=examples['train'];m=DelayedSparseCounts()
    phase(m,rows,'joint',training=True)
    assert m.changed_writes==3*len(rows) and 1<m.peak_pending<=6
    cp=m.checkpoint(); restored=DelayedSparseCounts.restore(cp)
    # New future dates avoid clock reversal while preserving weekday/clock keys.
    shifted=[replace(e,cutoff=e.cutoff+timedelta(days=7),due=e.due+timedelta(days=7)) for e in rows]
    assert phase(m,shifted,'joint',training=False)==phase(restored,shifted,'joint',training=False)
    assert m.changed_writes==cp['changed_writes']
