"""Long-source selection, bounded bytes, sealed fields and unchanged gates."""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import zlib
import pytest
from sara_engine.evaluation.casas_180_source import metadata, member_lines, timestamp
from sara_engine.evaluation.casas_180 import extract_lines, qualification
from sara_engine.evaluation.casas_longitudinal import Example, support
from sara_engine.utils.project_paths import workspace_path, ensure_parent_directory


def meta():
    return {'train_start':'2011-06-15','development_start':'2011-12-13',
            'test_start':'2012-02-11','clock_regression_days':[], 'valid_recorded_days':2071}


def test_date_only_selection_preserves_thirty_day_minimum_tail():
    first=datetime(2011,6,15)
    rows=[(first+timedelta(days=i)).strftime('%Y-%m-%d')+' 10:00:00 SENSOR_UNREAD MESSAGE_UNREAD' for i in range(180)]
    audit=metadata(rows)
    assert audit['train_recorded_days']==120
    assert audit['development_recorded_days']==audit['sealed_recorded_days']==30
    assert audit['valid_recorded_days']==180
    with pytest.raises(ValueError):metadata(rows[:-1])


def test_clock_invalid_dates_are_excluded_before_numeric_inspection():
    first=datetime(2011,6,15)
    rows=[(first+timedelta(days=i)).strftime('%Y-%m-%d')+' 10:00:00.0000 UNREAD UNREAD' for i in range(181)]
    rows.insert(1,rows[0].replace('10:00','09:00'))
    audit=metadata(rows)
    assert audit['clock_regression_days']==['2011-06-15']
    assert audit['valid_recorded_days']==180


def test_whitespace_sealed_fields_and_annotation_free_future():
    first=datetime(2011,6,15,10)
    rows=[]
    for i in range(25):
        t=(first+timedelta(seconds=60*i)).strftime('%Y-%m-%d %H:%M:%S')
        rows.append(t+' '+('KitchenA' if i%2 else 'Bedroom')+' ON')
    rows.append('2012-02-11 INVALID_TIME INVALID_SENSOR INVALID_MESSAGE')
    examples,audit=extract_lines(rows,meta())
    assert {e.label for e in examples['train']}=={0,1}
    assert audit['sealed_date_only_rows']==1
    assert all(300<=(e.due-e.cutoff).total_seconds()<=600 for e in examples['train'])


def test_cross_split_followup_and_history_purging():
    first=datetime(2011,12,12,23,50)
    rows=[(first+timedelta(seconds=60*i)).strftime('%Y-%m-%d %H:%M:%S')+' KitchenA ON' for i in range(30)]
    examples,audit=extract_lines(rows,meta())
    assert audit['cross_split_outcomes']>0
    assert all(e.cutoff.date()==e.due.date() for group in examples.values() for e in group)


def test_excluded_metadata_day_does_not_read_other_fields():
    m=meta();m['clock_regression_days']=['2011-06-16']
    examples,audit=extract_lines(['2011-06-16 BAD_TIME BAD_SENSOR BAD_MESSAGE'],m)
    assert audit['metadata_excluded_rows']==1 and not examples['train']


def test_deflate_integrity_and_size_fail_closed():
    raw=b'2011-06-15 10:00:00 KitchenA ON\n'
    encoder=zlib.compressobj(wbits=-15);compressed=encoder.compress(raw)+encoder.flush()
    path=Path(ensure_parent_directory(workspace_path('cache','tests','casas_180_member.deflate')))
    path.write_bytes(compressed)
    receipt={'compressed_size':len(compressed),'compressed_sha256':sha256(compressed).hexdigest(),
             'size':len(raw),'crc32':zlib.crc32(raw),'uncompressed_sha256':sha256(raw).hexdigest()}
    assert list(member_lines(path,receipt))==[raw.decode().rstrip()]
    with pytest.raises(ValueError):list(member_lines(path,{**receipt,'size':len(raw)-1}))
    with pytest.raises(ValueError):list(member_lines(path,{**receipt,'crc32':receipt['crc32']^1}))
    with pytest.raises(ValueError):list(member_lines(path,{**receipt,'compressed_sha256':'0'*64}))


def test_timestamp_variable_precision():
    assert timestamp('2011-06-15','10:00:00.0317').microsecond==31700
    assert timestamp('2011-06-15','10:00:00.031700000').microsecond==31700


def test_previous_support_thresholds_are_never_reduced():
    t=datetime(2011,6,15,10)
    e=Example(t,t+timedelta(seconds=300),(('KitchenA','ON'),)*8,(2,)*7,1)
    rows={'train':[e]*499,'development':[e]*199}
    assert not support(rows)['passed']
    assert not qualification(rows,meta())['passed']

def test_frozen_real_source_conditions_and_writes():
    from sara_engine.utils.project_paths import processed_data_path, project_path
    from sara_engine.evaluation.casas_180_source import file_sha256
    result=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_result_v1.json')).read_text())
    q=result['qualification']
    assert q['passed'] and q['contrast_groups']==73
    assert q['recurrence']['raw_repeat_row_fraction']==1
    assert q['splits']['development']['labels']=={'0':9196,'1':940}
    assert q['splits']['development']['class_days']=={'0':60,'1':60}
    comparison=result['comparison'];arms=comparison['arms']
    assert {row['changed_writes'] for row in arms.values()}=={56028}
    assert all(row['development_writes']==0 and row['saved_state_replay'] for row in arms.values())
    assert comparison['gates']=={'brier_gain':False,'balanced_gain':False,'gap_necessity':False,
                                 'paired_outcome':True,'scalar_equivalence':True}
    assert not comparison['quality_passed'] and not result['test_opened']
    for arm,h in result['model_sha256'].items():
        assert file_sha256(project_path('models','casas_hh103_180',arm+'_v1.json.gz'))==h


def test_registration_integrity_and_old_evidence_unchanged():
    from sara_engine.utils.project_paths import processed_data_path, project_path
    from sara_engine.evaluation.casas_180_source import file_sha256
    registration=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_registration_v1.json')).read_text())
    for p,h in {**registration['code'],**registration['preserved']}.items():
        assert file_sha256(project_path(p))==h
    protocol=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_protocol_v1.json')).read_text())
    assert file_sha256(processed_data_path('benchmark_fixtures','casas_180_protocol_v1.json'))==registration['protocol_sha256']
    assert 'no diagnostic bypass' in protocol['stop']


def test_metadata_exposure_and_expanded_source_hash_are_pinned():
    from sara_engine.utils.project_paths import processed_data_path
    meta=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_metadata_v1.json')).read_text())
    acquisition=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_acquisition_v1.json')).read_text())
    assert meta['valid_recorded_days']==2071 and meta['sealed_recorded_days']==1831
    assert meta['train_recorded_days']==180 and meta['development_recorded_days']==60
    assert len(acquisition['uncompressed_sha256'])==64 and acquisition['crc32']==532271806

def test_post_score_training_support_audit_does_not_change_counts():
    import gzip
    from sara_engine.utils.project_paths import model_path, processed_data_path
    audit=json.loads(Path(processed_data_path('benchmark_fixtures','casas_180_training_branch_audit_v1.json')).read_text())
    assert 'no tuning' in audit['scope']
    for arm,expected in audit['arms'].items():
        state=json.loads(gzip.decompress(Path(model_path('casas_hh103_180',arm+'_v1.json.gz')).read_bytes()))
        counts=[sum(v) for k,v in state['cells'].items() if k.startswith("('detail',")]
        assert expected['detail_routes']==len(counts)
        assert expected['training_outcomes']==sum(counts)==18676
        assert expected['singleton_routes']==sum(n==1 for n in counts)
