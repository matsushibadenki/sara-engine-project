"""Observed future notifications, complete controls, and frozen real receipts."""
from datetime import datetime, timedelta
import json
from pathlib import Path
from sara_engine.evaluation.casas_notification import extract_lines
from sara_engine.evaluation.motion_temporal import digest
from sara_engine.utils.project_paths import processed_data_path, project_path


def test_future_measurement_ignores_annotations_and_sealed_values():
    t=datetime(2011,6,15,10)
    lines=[]
    for i in range(25):
        stamp=(t+timedelta(seconds=60*i)).strftime('%Y-%m-%d,%H:%M:%S')
        sensor='KitchenA' if i%2 else 'Bedroom'
        lines.append(stamp+','+sensor+',ON,Never_Read_This_Annotation\n')
    lines.append('2011-07-30,INVALID_TIME,INVALID_SENSOR,INVALID_MESSAGE\n')
    examples,audit=extract_lines(lines)
    assert {e.label for e in examples['train']}=={0,1}
    assert audit['sealed_timestamp_only_rows']==1
    assert all(300<=(e.due-e.cutoff).total_seconds()<=600 for e in examples['train'])


def test_annotation_only_row_is_not_a_notification_outcome():
    t=datetime(2011,6,15,10)
    lines=[]
    for i in range(25):
        stamp=(t+timedelta(seconds=60*i)).strftime('%Y-%m-%d,%H:%M:%S')
        lines.append(stamp+',Bedroom,ON\n')
        if i==12:lines.append(stamp+',UNREAD_ANNOTATION\n')
    examples,audit=extract_lines(lines)
    assert audit['non_notification_rows']==1
    assert all(e.label==0 for e in examples['train'])


def test_split_crossings_are_purged():
    t=datetime(2011,7,14,23,50)
    lines=[(t+timedelta(seconds=60*i)).strftime('%Y-%m-%d,%H:%M:%S')+',KitchenA,ON\n' for i in range(30)]
    examples,audit=extract_lines(lines)
    assert audit['cross_split_outcomes']>0
    assert all(e.cutoff.date()==e.due.date() for rows in examples.values() for e in rows)


def test_registered_code_and_retained_negative_hashes():
    for name in ('casas_longitudinal_registration_v2.json','casas_notification_diagnostic_registration_v1.json'):
        reg=json.loads(Path(processed_data_path('benchmark_fixtures',name)).read_text())
        for p,h in {**reg['code'],**reg['preserved']}.items():assert digest(project_path(p))==h


def test_actual_real_changes_and_all_quality_failures_are_retained():
    result=json.loads(Path(processed_data_path('benchmark_fixtures','casas_notification_diagnostic_result_v1.json')).read_text())
    assert not result['known_source_support']['passed'] and not result['eligible_for_promotion']
    arms=result['comparison']['arms']
    assert len(arms)==8 and {row['changed_writes'] for row in arms.values()}=={18192}
    assert all(row['development_writes']==0 and row['saved_state_replay'] and row['peak_pending']<=32 for row in arms.values())
    assert result['comparison']['trace_sha256']['joint']==result['comparison']['trace_sha256']['scalar_joint']
    assert not result['comparison']['quality_passed'] and not result['test_opened']
    for arm,h in result['model_sha256'].items():
        assert digest(project_path('models','casas_hh102_diagnostic',arm+'_v1.json'))==h

def test_historical_code_matches_retained_registration_hashes():
    history=json.loads(Path(processed_data_path('benchmark_fixtures','casas_source_implementation_history_v1.json')).read_text())
    import hashlib
    hashes=set()
    for snapshot in history['snapshots'].values():
        assert hashlib.sha256(snapshot['source'].encode()).hexdigest()==snapshot['sha256']
        hashes.add(snapshot['sha256'])
    reg=json.loads(Path(processed_data_path('benchmark_fixtures','casas_longitudinal_registration_v1.json')).read_text())
    assert reg['code']['src/sara_engine/evaluation/casas_longitudinal.py'] in hashes
    assert reg['code']['scripts/eval/casas_longitudinal_comparison.py'] in hashes
    reg=json.loads(Path(processed_data_path('benchmark_fixtures','casas_notification_registration_v1.json')).read_text())
    assert reg['code']['src/sara_engine/evaluation/casas_notification.py'] in hashes


def test_variable_fractional_times_and_bad_clock_rejection():
    import pytest
    examples,_=extract_lines(['2011-06-15,10:43:05.0317,Bedroom,ON\n'])
    assert not examples['train']
    with pytest.raises(ValueError):
        extract_lines(['2011-06-15,10:43:05.0317,Bedroom,ON\n',
                       '2011-06-15,10:43:04.123,Bedroom,OFF\n'])
