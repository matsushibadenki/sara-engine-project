"""External model hooks and evidence-only diagnosis for capture protocol v3."""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from unittest.mock import patch

from sara_engine.evaluation.reactivation_codec import ReportCodec
from sara_engine.evaluation.route_numeric_v2 import reference_frame
from sara_engine.evaluation.route_snapshot_contract import (
    STATE_FIELDS, capture, compare_pair, diagnose, encode, window_support,
)

PROTOCOL_SHA256 = 'f6bbe9940c42d47ced0b846e6de2406b28f91aaae9592c8f558515eb7450d68e'


class Observer:
    """Keep every hook field outside model instances and evaluator labels."""

    def __init__(self, roster):
        self.roster=deepcopy(roster)
        self.by_tick={};self.finals={}
        self.snapshots={};self.actual_order=[];self.read_only_checks=[]
        self.current_case=None;self.installed=False
        for point in roster:
            key=(point['case_id'],point['arm'])
            if point['stage']=='final':self.finals[key]=point
            else:self.by_tick.setdefault((*key,point['tick']),[]).append(point)
        if len({encode(p) for p in roster})!=len(roster):raise ValueError('Duplicate registered hook')

    def set_case(self, identity):
        self.current_case=identity

    def selected(self, model):
        return model.strategy in ('error_triggered','permanent')

    def take(self, model, point, state_digest):
        key=encode(point)
        if key in self.snapshots:raise ValueError('Duplicate actual boundary')
        before=state_digest(model)
        value=capture({k:model.__dict__[k] for k in STATE_FIELDS},point,'future_external_state')
        after=state_digest(model)
        if before!=after:raise ValueError('Read-only capture changed complete model state')
        self.snapshots[key]=value
        self.actual_order.append(deepcopy(point))
        self.read_only_checks.append({'point':deepcopy(point),'before_sha256':before,'after_sha256':after})
        return value

    @contextmanager
    def install(self):
        from sara_engine.evaluation.sparse_route_reactivation import ReactivationMemory, state_digest
        if self.installed:raise ValueError('Nested hook installation forbidden')
        original_narrow,original_observe=ReactivationMemory.narrow,ReactivationMemory.observe
        observer=self

        def narrow(model):
            if not observer.selected(model):return original_narrow(model)
            identity=(observer.current_case,model.strategy,model.steps)
            planned=observer.by_tick.get(identity,[])
            if planned and (len(planned)!=2 or not planned[0]['stage'].startswith('pre_') or planned[1]['stage']!='post_'+planned[0]['stage'][4:]):
                raise ValueError('Invalid registered hook pair')
            prior=(model.pruned,len(model.starts),len(model.commits))
            before=observer.take(model,planned[0],state_digest) if planned else None
            original_narrow(model)
            events=[]
            if not prior[0] and model.pruned:events.append('initial')
            if len(model.starts)!=prior[1]:
                if len(model.starts)!=prior[1]+1:raise ValueError('Unexpected start increment')
                events.append('start')
            if len(model.commits)!=prior[2]:
                if len(model.commits)!=prior[2]+1:raise ValueError('Unexpected commit increment')
                events.append('commit')
            expected=[planned[0]['stage'][4:]] if planned else []
            if events!=expected:raise ValueError('Actual event differs from finite registered boundary plan')
            if planned:
                after=observer.take(model,planned[1],state_digest)
                compare_pair(before,after)

        def observe(model,outcome,time):
            original_observe(model,outcome,time)
            if observer.selected(model) and model.steps==1024:
                point=observer.finals.get((observer.current_case,model.strategy))
                if point is None:raise ValueError('Unregistered final state')
                observer.take(model,point,state_digest)

        self.installed=True
        try:
            with patch.object(ReactivationMemory,'narrow',narrow),patch.object(ReactivationMemory,'observe',observe):
                yield self
        finally:
            self.installed=False

    def finish(self):
        expected={encode(p) for p in self.roster}
        if set(self.snapshots)!=expected or len(self.actual_order)!=len(self.roster):
            raise ValueError('Missing/extra actual boundary evidence')
        return [self.snapshots[encode(p)] for p in self.roster]


def record(payload,saved,context):
    codec=ReportCodec(saved)
    value=codec.normalize(payload)
    reference,mapping=reference_frame(saved,context['registration_sha256'],context['preserved_count'])
    return {'schema':'sara-route-capture-numeric-report-v3','phase':context['phase'],
        'selection_sha256':PROTOCOL_SHA256,'registration_sha256':context['registration_sha256'],
        'execution_sha256':context['execution_sha256'],'runtime':context['runtime'],
        'payload':value,'reference_metadata_mapping':mapping,
        'payload_canonical_sha256':sha256(codec.encode(value)).hexdigest(),
        'historical_values_exact_match':codec.equivalent(reference,value),
        'old_failed_replay_reclassified':False,'conditions_relaxed':False,'test_opened':False}


def validate_record(value,saved,context):
    if encode(record(value['payload'],saved,context))!=encode(value):raise ValueError('Numerical report provenance/content differs')
    if value['historical_values_exact_match'] is not True or value['payload']['execution_integrity_passed'] is not True:
        raise ValueError('Unchanged numerical reference/integrity gate failed')
    return value


def evaluator_truth(cases):
    """Called after all case learning ends, never from a hook or learner."""
    from sara_engine.evaluation.sparse_route_reactivation import dependencies, ROUTES
    result={}
    for case in cases:
        identity=case['task']+':'+str(case['seed'])+':'+case['condition']
        result[identity]={
            'initial':sorted(ROUTES.index(r) for r in dependencies(case['seed'],case['task'],False)),
            'final':sorted(ROUTES.index(r) for r in dependencies(case['seed'],case['task'],case['condition']=='changed'))}
    return result


def analyze(collection,payload,truth):
    """Only supplied new snapshots and separately supplied evaluator IDs."""
    snapshots=collection['snapshots']
    index={(s['point']['case_id'],s['point']['arm'],s['point']['tick'],s['point']['stage']):s for s in snapshots}
    rows=[];missing_commit=[];missing_final=[];support_failures=0;windows=0
    for case in payload['cases']:
        identity=case['task']+':'+str(case['seed'])+':'+case['condition']
        labels=truth[identity]
        for arm in ('error_triggered','permanent'):
            initial=index[(identity,arm,180,'post_initial')]
            final=index[(identity,arm,1024,'final')]
            initial_diagnostic=diagnose(initial,labels['initial'])
            final_diagnostic=diagnose(final,labels['final'])
            retained=all(d['retained'] for d in final_diagnostic['dependencies'])
            stats=case['arms'][arm]
            if retained is not stats['final_dependencies_retained'] or any(i not in initial['pool'] for i in labels['final']) is not stats['initial_new_dependency_absent']:
                raise ValueError('Snapshot/evaluator membership differs from numerical report')
            item={'case_id':identity,'task':case['task'],'seed':case['seed'],'condition':case['condition'],'arm':arm,
                'initial_diagnostic':initial_diagnostic,'final_diagnostic':final_diagnostic,
                'reported_final_retention':retained,'tail_exact':stats['tail_exact'],
                'frozen_exact':stats['frozen']['exact_accuracy'],'exploration_windows':[]}
            if arm=='error_triggered' and case['condition']=='changed':
                for d in final_diagnostic['dependencies']:
                    if not d['retained']:missing_final.append({'case_id':identity,**d})
            for commit in stats['commits']:
                tick=commit['tick'];start_tick=tick-180
                start=index[(identity,arm,start_tick,'pre_start')]
                pre=index[(identity,arm,tick,'pre_commit')]
                post=index[(identity,arm,tick,'post_commit')]
                labels_now=labels['final'] if tick>=512 else labels['initial']
                diagnostic=diagnose(post,labels_now)
                window=window_support(start,pre);windows+=1
                support=[]
                for d in diagnostic['dependencies']:
                    rid=d['route_id'];w=window['route_window_rows'][rid]
                    key_support=[w[3][j]+w[3][j+1] for j in range(0,len(w[3]),2)]
                    support.append({'route_id':rid,'window_observations':w[0],
                        'window_key_support':key_support,'window_label_counts':w[3],
                        'window_error_ratio':w[1]/w[2],
                        'ranked_in_window_top_three':rid in window['window_ranked_pool']})
                    if not d['support_ready']:support_failures+=1
                    if not d['retained']:missing_commit.append({'case_id':identity,**d,
                        'window_observations':w[0],'window_key_support':key_support,
                        'ranked_in_window_top_three':rid in window['window_ranked_pool']})
                item['exploration_windows'].append({'start':start_tick,'commit':tick,
                    'post_commit_diagnostic':diagnostic,'dependency_window_support':support,
                    'all_route_window_observations':[r[0] for r in window['route_window_rows']],
                    'window_ranked_pool':window['window_ranked_pool'],
                    'cumulative_ranked_pool':window['current_cumulative_ranked_pool'],
                    'counterfactual_accuracy':None,'causal_attribution':None})
            rows.append(item)
    failed_cases=sorted({d['case_id'] for d in missing_final})
    return {'schema':'sara-route-capture-analysis-v3','truth_provenance':'Pinned evaluator invoked only after all case learning; never learner/hook input',
        'evaluator_truth':truth,'case_arm_diagnostics':rows,
        'missing_at_commit':missing_commit,'missing_at_final':missing_final,
        'summary':{'snapshot_count':len(snapshots),'diagnosed_case_arms':len(rows),'exploration_windows':windows,
            'commit_dependency_support_below16':support_failures,
            'final_candidate_retention_failure_cases':failed_cases,
            'missing_dependency_count_at_commit':len(missing_commit),
            'commit_exclusions_by_score':sum(d['rank_exclusion_kind']=='score' for d in missing_commit),
            'commit_exclusions_by_tie_break':sum(d['rank_exclusion_kind']=='tie_break' for d in missing_commit),
            'missing_at_commit_but_window_top_three':sum(d['ranked_in_window_top_three'] for d in missing_commit)},
        'causal_attribution':None,'counterfactual_accuracy':None,'functional_substitution_proven':None,
        'component_passed':payload['component_passed'],'conditions_relaxed':False,'test_opened':False,
        'old_failed_replay_reclassified':False,'old_discarded_trace_equality':None,'promotion_authorized':False}
