"""Bounded plain-data encoder checks; no model, host or candidate capture calls."""
from dataclasses import replace
from hashlib import sha256
import json
from math import isfinite
from unittest.mock import patch
import sys
from sara_engine.evaluation import route_capture_candidate as v1
from sara_engine.evaluation import route_capture_candidate_v2 as v2
from sara_engine.evaluation.route_capture_layout import decode,validate,canonical
from sara_engine.evaluation.route_capture_layout_preflight import fixture,hand_encoding

SELECTION_SHA='68ddc0f9f0510688c67e827a86f37fbf50daea8c6918b6d7f4d84c0e8edd8029'


class EncoderBoundError(ValueError):
    """Plain-data shape or encoding exceeds this verification wrapper's scope."""


def require(value,message):
    if not value:raise EncoderBoundError(message)


def bounded_encode(value,*,depth_limit=24,node_limit=20000,byte_limit=262144,counter=None):
    """Guard only this plain-data verification call, not the dormant candidate."""
    remaining=node_limit;ancestors=set()
    def check(item,depth):
        nonlocal remaining
        remaining-=1
        require(remaining>=0 and depth<=depth_limit,'Plain-data depth/node limit exceeded')
        kind=type(item)
        if item is None or kind in (bool,int,str):return
        if kind is float:
            require(isfinite(item),'Nonfinite float');return
        require(kind in (tuple,list,dict),'Unsupported plain-data type')
        identity=id(item)
        require(identity not in ancestors,'Cyclic plain data')
        ancestors.add(identity)
        try:
            if kind is dict:
                for key,entry in item.items():check(key,depth+1);check(entry,depth+1)
            else:
                for entry in item:check(entry,depth+1)
        finally:ancestors.remove(identity)
    check(value,0)
    if counter is not None:counter[0]+=1
    encoded=v1._primitive(value)
    raw=v1._encoded(encoded)
    require(len(raw)<=byte_limit,'Encoded byte limit exceeded')
    return encoded,raw


def reordered(value):
    if type(value) is dict:return {reordered(k):reordered(v) for k,v in reversed(list(value.items()))}
    if type(value) is list:return [reordered(item) for item in value]
    if type(value) is tuple:return tuple(reordered(item) for item in value)
    return value


def faults(spec,counter):
    cycle=[];cycle.append(cycle)
    operations=[('unsupported_object',lambda:bounded_encode(object(),counter=counter)),
                ('nonfinite_float',lambda:bounded_encode(float('nan'),counter=counter)),
                ('cycle',lambda:bounded_encode(cycle,counter=counter)),
                ('depth_limit',lambda:bounded_encode([[[None]]],depth_limit=1,counter=counter)),
                ('node_limit',lambda:bounded_encode([None,None],node_limit=2,counter=counter)),
                ('byte_limit',lambda:bounded_encode([],byte_limit=1,counter=counter))]
    rejected=[]
    for name,operation in operations:
        try:operation()
        except EncoderBoundError:rejected.append(name)
        else:raise ValueError('Undetected plain-data fault: '+name)
    require(rejected==spec['protocol']['faults'],'Fault roster changed')
    return rejected


def analyze(spec,sources,reports):
    require(set(sources)==set(spec['sources']) and len(sources)==7,'Source roster differs')
    require(set(reports)==set(spec['reports']),'Report roster differs')
    require(sum(len(x) for x in (*sources.values(),*reports.values()))<=spec['bounds']['input_bytes'],'Input budget exceeded')
    for pins,inputs in ((spec['sources'],sources),(spec['reports'],reports)):
        for name,raw in inputs.items():require(sha256(raw).hexdigest()==pins[name],'Frozen input changed')
    parent=json.loads(reports['route_capture_candidate_v2_result_v1.json'])
    require(parent['launch_authorized'] is False and parent['real_model_constructed'] is False,'Candidate promoted')
    require(v1.LAUNCH_AUTHORIZED is False and v2.LAUNCH_AUTHORIZED is False,'Candidate launch flag changed')
    before_modules=set(sys.modules);records=[];pair_checks=[];counter=[0]
    # Reuse plain hand recipes only. Traps prevent accidentally crossing a model/host boundary.
    with patch.object(v1.RouteCaptureCandidate,'__init__',side_effect=AssertionError('No model constructor')), \
         patch.object(v2.RouteCaptureCandidateV2,'__init__',side_effect=AssertionError('No model constructor')), \
         patch.object(v1.RouteCaptureCandidate,'capture',side_effect=AssertionError('No candidate capture')), \
         patch.object(v2.RouteCaptureCandidateV2,'capture',side_effect=AssertionError('No candidate capture')), \
         patch.object(v1.OwnerHost,'create',side_effect=AssertionError('No host creation')):
        for variant in spec['variants']:
            pair,pre,post=fixture(spec,variant);built={}
            for phase_name,state in (('predecessor',pre),('receipt',post)):
                expected=hand_encoding(state);original=canonical(expected)
                for order in spec['protocol']['order_variants']:
                    value=state if order=='original' else reordered(state)
                    encoded,raw=bounded_encode(value,depth_limit=spec['bounds']['data_depth'],
                                               node_limit=spec['bounds']['data_nodes'],byte_limit=spec['bounds']['encoded_bytes'],counter=counter)
                    require(encoded==expected and raw==original,'Encoder/reference differ')
                    require(hand_encoding(decode(encoded))==expected,'Decoder changed exact types/values')
                    require(canonical(hand_encoding(state))==original,'Encoder changed input state')
                    built[(phase_name,order)]=replace(getattr(pair,phase_name),state=encoded,digest=sha256(raw).hexdigest())
                    records.append({'variant':variant,'phase':phase_name,'order':order,'encoded_bytes':len(raw),
                                    'sha256':sha256(raw).hexdigest(),'independent_reference_exact':True,'typed_roundtrip_exact':True})
            for order in spec['protocol']['order_variants']:
                actual=replace(pair,predecessor=built[('predecessor',order)],receipt=built[('receipt',order)])
                projected=validate(actual)
                require(projected['learning_fields']==435 and projected['allocation_flags']==150,'Logical projection differs')
                pair_checks.append({'variant':variant,'order':order,'fields':435,'allocations':150,'capture_unblocked':False})
        require(counter[0]==spec['protocol']['baseline_calls']==8,'Encoder baseline count differs')
        negative_before=counter[0];rejected=faults(spec,counter)
        require(counter[0]-negative_before==1,'Negative encoder call count differs')
    require(set(sys.modules)==before_modules,'Model/helper module imported by plain encoder verification')
    return {'schema':'route-encoder-check-result-v1','records':records,'pair_projection_checks':pair_checks,
            'baseline_encoder_calls':8,'negative_encoder_calls':1,'faults_rejected':rejected,
            'independent_reference_exact':True,'missing_runtime_evidence':parent['missing_runtime_evidence'],
            'scope':'Original pure format encoder on two existing hand recipes only; no candidate body, model or live snapshot.',
            'real_model_constructed':False,'host_created':False,'candidate_capture_executions':0,'live_numeric_capture_executions':0,
            'learner_executions':0,'prediction_error_calculator_executions':0,'model_storage_executions':0,
            'task_scores':0,'acquisitions':0,'resource_measurements':0,'capture_unblocked':False,
            'launch_authorized':False,'snn_advantage_established':False,'conditions_relaxed':False,'test_opened':False}
