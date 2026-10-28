"""Pure saved-evidence audit; deliberately imports no learner or evaluator."""
from itertools import combinations, product
import json

CELLS = tuple(h+'__'+e for h in ('cumulative','discounted') for e in ('uniform','public_exponential'))
ROUTES = tuple((key, output) for width in (1,2) for key in combinations(range(5),width) for output in range(3))
BOUNDARIES = (180,512,768,1024)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def validate_snapshot(snapshot, history, boundary):
    require(snapshot['boundary']==boundary and snapshot['query_tick']==min(boundary,1023) and snapshot['read_only'] is True,'Snapshot boundary/read-only contract differs')
    routes=snapshot['routes']
    require(len(routes)==45 and [r['route'] for r in routes]==list(range(45)),'Route catalog differs')
    require(sum(r['observations'] for r in routes)==9*boundary,'Observation conservation failed')
    require(sum(r['in_pool'] for r in routes)==(45 if boundary==180 else 9),'Pool capacity differs')
    true_ids=[r['route'] for r in routes if r['true_dependency']]
    require(len(true_ids)==3 and sorted(ROUTES[i][1] for i in true_ids)==[0,1,2],'Dependency flags differ')
    scale=2**(min(boundary,1023)//64) if history=='discounted' else 1
    for output in range(3):
        group=[r for r in routes if ROUTES[r['route']][1]==output]
        ranked=sorted(group,key=lambda r:(r['score'],r['route']))
        require([r['rank'] for r in ranked]==list(range(1,16)),'Saved rank disagrees with original score ordering')
    for route in routes:
        require(type(route['observations']) is int and route['observations']>=0,'Invalid observation count')
        require(type(route['score']) is float and route['score']>=0,'Invalid score')
        inputs,_=ROUTES[route['route']]
        require([k['key'] for k in route['keys']]==[list(k) for k in product((0,1),repeat=len(inputs))],'Key catalog differs')
        for key in route['keys']:
            mass=key['raw_mass']
            if mass is None:
                require(key['probability'] is None and key['effective_mass'] is None,'Absent key imputed')
            else:
                require(len(mass)==2 and all(type(n) is int and n>=0 for n in mass),'Invalid mass')
                require(key['effective_mass']==[n/scale for n in mass],'Effective mass differs')
                require(key['probability']==(mass[1]+scale)/(sum(mass)+2*scale),'Stored probability differs from registered rule')
    return true_ids


def boundary_diagnostic(snapshot):
    routes=snapshot['routes']
    selected=[]
    for slot in range(3):
        eligible=[r for r in routes if ROUTES[r['route']][1]==slot and r['in_pool'] and r['observations']>=16]
        selected.append(min(eligible,key=lambda r:(r['score'],r['route']))['route'] if eligible else None)
    return {'boundary':snapshot['boundary'],'eligible_readout_routes':selected,
            'true_dependencies':[{'route':r['route'],'rank':r['rank'],'observations':r['observations'],
               'in_pool':r['in_pool'],'support_eligible':r['observations']>=16,
               'selected_for_output':r['route'] in selected,
               'probabilities':[k['probability'] for k in r['keys']]} for r in routes if r['true_dependency']]}


def pair_at_boundary(left,right):
    differences={'observations':[],'rank':[],'pool':[],'probability':[],'equal_mass_different_probability':[]}
    for a,b in zip(left['routes'],right['routes']):
        i=a['route']
        if a['observations']!=b['observations']:differences['observations'].append(i)
        if a['rank']!=b['rank']:differences['rank'].append(i)
        if a['in_pool']!=b['in_pool']:differences['pool'].append(i)
        for ka,kb in zip(a['keys'],b['keys']):
            if ka['probability']!=kb['probability']:
                differences['probability'].append([i,ka['key'],ka['probability'],kb['probability']])
                if ka['raw_mass']==kb['raw_mass']:
                    differences['equal_mass_different_probability'].append([i,ka['key']])
    return {'boundary':left['boundary'],'different_route_observation_count':len(differences['observations']),
            'differences':differences}


def analyze(report):
    cases=report['cases']
    require(len(cases)==20,'Case budget differs')
    expected={(s,t,c) for s in (1201,1213,1229,1237,1249) for t in ('independent','interacting') for c in ('stationary','changed')}
    require({(c['seed'],c['task'],c['condition']) for c in cases}==expected,'Case identities differ')
    results=[]
    for case in cases:
        require(set(case['cells'])==set(CELLS),'Factor cells differ')
        diagnostics={}
        truth=None
        for name,cell in case['cells'].items():
            history=name.split('__')[0]
            stats=cell['training']
            require(stats['route_updates']==9216 and stats['actual_learning_field_changes']==36864,'Actual updates differ')
            require([s['boundary'] for s in cell['snapshots']]==list(BOUNDARIES),'Boundary roster differs')
            for s,b in zip(cell['snapshots'],BOUNDARIES):
                ids=validate_snapshot(s,history,b)
                if truth is None:truth=ids
                require(ids==truth,'Dependency identities vary across cells/boundaries')
            require(stats['final_dependencies_retained']==all(cell['snapshots'][-1]['routes'][i]['in_pool'] for i in truth),'Final retention mismatch')
            blocks=stats['blocks']
            require(len(blocks)==16 and [b['end'] for b in blocks]==list(range(64,1025,64)),'Recovery blocks differ')
            post=[b for b in blocks if b['end']>512]
            recovered=next((b['end']-512 for a,b in zip(post,post[1:]) if a['exact']>=.90 and b['exact']>=.90),None)
            require(stats['recovery_ticks']==recovered,'Recovery definition changed')
            intervals=[]
            for a,b in zip(cell['snapshots'],cell['snapshots'][1:]):
                increments=[y['observations']-x['observations'] for x,y in zip(a['routes'],b['routes'])]
                require(all(0<=v<=b['boundary']-a['boundary'] for v in increments),'Invalid route allocation increment')
                require(sum(increments)==9*(b['boundary']-a['boundary']),'Interval update budget differs')
                intervals.append({'start':a['boundary'],'end':b['boundary'],'route_increments':increments,
                                  'true_dependency_updates':sum(increments[i] for i in truth),
                                  'true_dependency_update_share':sum(increments[i] for i in truth)/sum(increments),
                                  'min_route_updates':min(increments),'max_route_updates':max(increments)})
            diagnostics[name]={'boundaries':[boundary_diagnostic(s) for s in cell['snapshots']],
                               'intervals':intervals,'recovery_ticks':recovered,
                               'recovery_within_256':recovered is not None and recovered<=256,
                               'exploration_starts':stats['exploration_starts'],'commits':stats['commits'],
                               'postchange_blocks':post}
        pairs=[]
        for left,right in combinations(CELLS,2):
            boundaries=[pair_at_boundary(a,b) for a,b in zip(case['cells'][left]['snapshots'],case['cells'][right]['snapshots'])]
            first={k:next((b['boundary'] for b in boundaries if b['differences'][k]),None) for k in ('observations','probability','rank','pool')}
            pairs.append({'left':left,'right':right,'first_saved_difference':first,'boundaries':boundaries})
        results.append({'seed':case['seed'],'task':case['task'],'condition':case['condition'],'cells':diagnostics,'pairs':pairs})
    return {'schema':'sara-route-factorial-audit-result-v1','cases':results,
            'summary':{'case_count':20,'cell_count':80,'snapshots':320,'route_key_cells':48000,
                       'intervals':240,'paired_boundary_comparisons':480,
                       'learner_executions':0,'new_prediction_scoring':0,'new_snapshots':0,
                       'test_opened':False,'causal_direct_effect_identified':False,
                       'snn_advantage_tested':False,'promotion_authorized':False}}
