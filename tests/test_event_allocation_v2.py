"""V2 orchestration/provenance checks; mock workers never fabricate evidence."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sara_engine.evaluation import event_allocation_v2 as audit
from sara_engine.evaluation import allocation_report as codec
from sara_engine.evaluation.allocation_report_preflight import constructed_payload
from scripts.eval import event_allocation_v2 as runner


def selection():
    return runner.read('selection')


def test_driver_dispatches_exact_fresh_roster_and_only_new_cli(monkeypatch):
    payload=constructed_payload()
    goldens=audit.inputs(selection())
    calls=[]
    def fake_run(command,**kwargs):
        index=int(command[command.index('--case')+1])
        arm=command[command.index('--arm')+1]
        row=deepcopy(payload['workers'][index*3+codec.MODES.index(arm)])
        row['receipt_sha256']=goldens[index][1]['semantic_trace_sha256']
        calls.append((command,kwargs))
        return SimpleNamespace(stdout=json.dumps(row))
    monkeypatch.setattr(audit.subprocess,'run',fake_run)
    monkeypatch.setattr(audit.frozen,'witnesses',lambda:payload['witnesses'])
    workers,witnesses=audit.collect(selection(),'execute')
    assert len(workers)==len(calls)==144 and witnesses==payload['witnesses']
    for command,kwargs in calls:
        assert command[1].endswith('/scripts/eval/event_allocation_v2.py')
        assert kwargs['env']['PYTHONHASHSEED']=='0'
        assert kwargs['timeout']<=30 and kwargs['check'] is True
    assert sum(w['shared_input_calls'] for w in workers if w['mode']=='neuron')==224
    assert sum(w['shared_input_calls'] for w in workers if w['mode']=='scalar')==224


@pytest.mark.parametrize('kind',['identity','site','output','process'])
def test_driver_failure_stops_without_more_workers(monkeypatch,kind):
    row=constructed_payload()['workers'][0]
    row['receipt_sha256']=audit.inputs(selection())[0][1]['semantic_trace_sha256']
    calls=[]
    def fake_run(*args,**kwargs):
        calls.append(1)
        if kind=='process':raise RuntimeError('worker failed')
        bad=deepcopy(row)
        if kind=='identity':bad['member']=1
        if kind=='site':
            bad['records'][0]['allocation']['post_query_live_sites']['x'*300]={'owner':'other','bytes':1,'blocks':1}
        return SimpleNamespace(stdout=' '*(262145) if kind=='output' else json.dumps(bad))
    monkeypatch.setattr(audit.subprocess,'run',fake_run)
    monkeypatch.setattr(audit.frozen,'witnesses',lambda:constructed_payload()['witnesses'])
    with pytest.raises((ValueError,RuntimeError)):
        audit.collect(selection(),'execute')
    assert calls==[1]


def test_v2_real_payload_round_trip_keeps_registration_and_failure():
    payload=constructed_payload()
    value=audit.payload(payload['workers'],payload['witnesses'],'1'*64,'2'*64,runner.runtime())
    assert not value['metadata']['synthetic']
    assert value['metadata']['registration_sha256']=='1'*64
    assert value['metadata']['execution_sha256']=='2'*64
    assert codec.decode(codec.encode(value))==value


def test_replication_requires_exact_owned_state_but_retains_variable_raw_bytes():
    original=constructed_payload()
    replica=deepcopy(original)
    replica['workers'][0]['records'][0]['allocation'][codec.SCALARS[3]]+=1
    receipt=audit.summarize(original,replica)
    assert receipt['replication']['semantic_and_ownership_exact_match']
    assert receipt['replication']['byte_identical_windows']==receipt['replication']['compared_windows']-1
    replica['workers'][0]['records'][0]['ownership']['output_only_bytes']+=1
    with pytest.raises(ValueError,match='semantic/ownership'):
        audit.summarize(original,replica)


def test_missing_saved_result_cannot_create_replication_marker_or_launch(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'verify',lambda **kwargs:None)
    monkeypatch.setattr(runner,'path',lambda s:tmp_path/(s+'.json'))
    def forbidden(*args):pytest.fail('Workers launched without saved result')
    monkeypatch.setattr(audit,'collect',forbidden)
    with pytest.raises(FileNotFoundError):runner.main('replay')
    assert not (tmp_path/'replay_execution.json').exists()


def test_synthetic_saved_result_cannot_launch_replication(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'verify',lambda **kwargs:None)
    monkeypatch.setattr(runner,'path',lambda s:tmp_path/(s+'.json'))
    for name in ('registration','execution'):(tmp_path/(name+'.json')).write_text('{}')
    value=constructed_payload()
    value['metadata']['registration_sha256']=runner.digest(runner.path('registration'))
    value['metadata']['execution_sha256']=runner.digest(runner.path('execution'))
    runner.path('result').write_bytes(codec.encode(value))
    monkeypatch.setattr(audit,'collect',lambda *args:pytest.fail('Synthetic result launched workers'))
    with pytest.raises(ValueError,match='synthetic'):runner.main('replay')
    assert not runner.path('replay_execution').exists()


def test_preserved_conditions_and_consumed_reports(monkeypatch):
    current=selection()
    assert len(current['preserved'])==378
    old=json.loads(Path(runner.path('selection').parent/'event_allocation_selection_v1.json').read_text())
    assert current['fixed_original_task']==old['fixed_original_task']
    assert current['measurement']==old['measurement']
    if not runner.path('registration').exists():pytest.skip('Code not yet frozen')
    runner.verify()
    if not runner.path('result').exists():pytest.skip('Primary finite execution not yet consumed')
    original=codec.decode(runner.path('result').read_bytes())
    assert original['metadata']['execution_sha256']==runner.digest(runner.path('execution'))
    monkeypatch.setattr(audit,'collect',lambda *args:pytest.fail('Consumed execution restarted'))
    with pytest.raises(FileExistsError):runner.main('execute')
    if runner.path('replay').exists():
        runner.main('replay')
        assert runner.read('replay')['replication']['semantic_and_ownership_exact_match']
