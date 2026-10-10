"""Causal receipt, sealed field and matched local-readout regression checks."""
import csv
from datetime import datetime, timedelta
from io import BytesIO, StringIO
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile
import pytest
from sara_engine.evaluation.motion_temporal import LocalReadout, paths, digest
from sara_engine.evaluation.room_temporal import keys, extract, contrast_audit, CODE
from sara_engine.utils.project_paths import project_path, processed_data_path

spec=importlib.util.spec_from_file_location('room_identification',project_path('scripts/eval/room_local_identification.py'))
diagnostic=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)

def test_due_validation_precedes_outcome_access():
    model=LocalReadout(); t=datetime(2017,12,22)
    model.predict(('a','b'),t,t+timedelta(seconds=60))
    with pytest.raises(ValueError):model.deliver(t,lambda:pytest.fail('Premature outcome access'),True)
    model.deliver(t+timedelta(seconds=60),lambda:1,True)
    assert model.writes==2 and model.cells=={'a':[0,1],'b':[0,1]}

def test_overlapping_receipts_do_not_learn_future(monkeypatch):
    observed=[]
    original=LocalReadout.predict
    def predict(self,*args):
        observed.append(self.writes)
        return original(self,*args)
    monkeypatch.setattr(LocalReadout,'predict',predict)
    t=datetime(2017,12,22)
    rows=[(t+timedelta(seconds=i*30),t+timedelta(seconds=i*30+60),((0,0),)*16,1) for i in range(3)]
    model,n=diagnostic.learn(rows,'age')
    assert observed==[0,0,2]
    assert model.writes==6 and n==3 and model.pending is None

def test_frozen_delivery_changes_no_counts():
    model=LocalReadout();t=datetime(2017,12,22)
    model.predict(('a','b'),t,t+timedelta(seconds=60));model.deliver(t+timedelta(seconds=60),lambda:0,False)
    assert not model.cells and model.writes==0

def test_same_current_distinct_age_retains_shared_route():
    t=datetime(2017,12,22,11)
    left=((1,0),)*15+((0,0),)
    right=((0,0),)*16
    assert keys(left,t,'current')==keys(right,t,'current')
    assert keys(left,t,'age')[0]==keys(right,t,'age')[0]
    assert keys(left,t,'age')[1]!=keys(right,t,'age')[1]
    assert keys(left,t,'scalar_age')==keys(left,t,'age')

def test_sealed_numeric_fields_are_not_parsed():
    stream=StringIO();writer=csv.writer(stream)
    writer.writerow(('Date','Time','S6_PIR','S7_PIR','Room_Occupancy_Count'))
    t=datetime(2017,12,22)
    for i in range(20):writer.writerow(((t+timedelta(seconds=30*i)).strftime('%Y/%m/%d'),(t+timedelta(seconds=30*i)).strftime('%H:%M:%S'),0,0,0))
    writer.writerow(('2018/01/10','00:00:00','SEALED','SEALED','SEALED'))
    archive=BytesIO()
    with ZipFile(archive,'w') as z:z.writestr('Occupancy_Estimation.csv',stream.getvalue())
    archive.seek(0); rows,audit=extract(archive)
    assert audit=={'sealed_timestamp_only_rows':1,'numeric_opened_rows':20}
    assert len(rows['train'])==3
    assert all((r[1]-r[0]).total_seconds()==60 for r in rows['train'])

def test_single_class_support_fails():
    assert not contrast_audit({'train':[],'development':[]})['passed']

def test_retained_code_and_negative_hashes():
    for prefix in ('room_temporal','room_local_identification'):
        reg=json.loads(Path(processed_data_path('benchmark_fixtures',prefix+'_registration_v1.json')).read_text())
        for p,h in {**reg['code'],**reg.get('preserved',{})}.items():assert digest(project_path(p))==h

def test_real_diagnostic_writes_and_no_quality_scores():
    report=json.loads(Path(processed_data_path('benchmark_fixtures','room_local_identification_result_v1.json')).read_text())
    assert {r['changed_writes'] for r in report['arms'].values()}=={10400}
    assert report['learned_same_current_opposite_posterior_groups']==4
    assert report['scalar_equivalent'] and not report['development_scores_computed'] and not report['test_opened']

def test_canonical_real_support_replay():
    spec=importlib.util.spec_from_file_location('room_replay',project_path('scripts/eval/replay_room_support.py'))
    replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
    replay.main()
