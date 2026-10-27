"""Prospective bounded, external, read-only route evidence contract."""
from copy import deepcopy
from hashlib import sha256
from itertools import combinations, product
import json
import math

PROTOCOL_SHA256 = '04ceb20739961386c075ecbbff6d110ae3a3eb50ce4fb8b28e5ffb028b803663'
LIMITS = {'snapshots':160,'routes':45,'keys_per_route':4,'tick_max':1024,
          'bytes_per_snapshot':8192,'header_bytes':65536,'report_bytes':2097152,
          'static_artifact_bytes':262144,'static_total_bytes':1048576}
STAGES = ('pre_initial','post_initial','pre_start','post_start','pre_commit','post_commit','final')
CATALOG = [{'inputs':list(key),'output':out} for width in (1,2)
           for key in combinations(range(5),width) for out in range(3)]
STATE_FIELDS = {'routes','tables','errors','weights','observations','steps','pending','pool'}


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def integer(value,low=0,high=1024):
    if type(value) is not int or not low<=value<=high: raise ValueError('Bounded integer required')
    return value


def number(value):
    if type(value) is not float or not math.isfinite(value): raise ValueError('Finite float required')
    return value


def pool(value,size=None):
    if type(value) is not list or any(type(i) is not int or not 0<=i<45 for i in value):
        raise ValueError('Bounded route ID list required')
    if value!=sorted(set(value)) or (size is not None and len(value)!=size):raise ValueError('Invalid route pool')
    return value


def point(value):
    if type(value) is not dict or set(value)!={'case_id','arm','tick','stage'}:raise ValueError('Invalid boundary identity')
    text=value['case_id']
    if type(text) is not str or not 1<=len(text)<=64 or any(not 32<=ord(c)<127 for c in text):raise ValueError('Bounded printable case identity required')
    if type(value['arm']) is not str or type(value['stage']) is not str:raise ValueError('Exact string boundary tags required')
    if value['arm'] not in ('error_triggered','permanent') or value['stage'] not in STAGES:raise ValueError('Unregistered boundary kind/arm')
    integer(value['tick'])
    if value['stage']=='final' and value['tick']!=1024:raise ValueError('Final boundary must follow all outcomes')
    if value['stage'].endswith('initial') and value['tick']!=180:raise ValueError('Original initial boundary changed')
    return value


def global_weight(tick):
    q,r=divmod(tick,64)
    return 9*(64*((1<<q)-1)+r*(1<<q))


def validate_rows(rows,tick):
    if type(rows) is not list or len(rows)!=45:raise ValueError('All 45 candidate rows are required')
    for row,route in zip(rows,CATALOG):
        if type(row) is not list or len(row)!=4:raise ValueError('Expected count/error/weight/key-count vector')
        count,error,weight,vector=row
        integer(count,0,tick)
        number(error);number(weight)
        if type(vector) is not list or len(vector)!=2*(2**len(route['inputs'])):raise ValueError('Complete canonical key vector required')
        for n in vector:integer(n)
        if sum(vector)!=count:raise ValueError('Class counts and observations disagree')
        if count==0:
            if error!=0.0 or weight!=0.0:raise ValueError('Unobserved route has fabricated statistics')
        elif not (0.0<error<weight and count<=weight<=count*(1<<max(0,(tick-1)//64))):
            raise ValueError('Weighted local statistics outside fixed finite bounds')
    if sum(row[0] for row in rows)!=9*tick or sum(row[2] for row in rows)!=global_weight(tick):
        raise ValueError('Global actual-observation/public-time weight conservation failed')
    for out in range(3):
        group=[row for row,route in zip(rows,CATALOG) if route['output']==out]
        if sum(r[0] for r in group)!=3*tick or sum(r[2] for r in group)!=global_weight(tick)/3:
            raise ValueError('Per-output conserved update budget failed')


def costs(rows):
    return [(row[1]/row[2] if row[2] else 0.25)+0.005*len(route['inputs'])
            for row,route in zip(rows,CATALOG)]


def ranked_pool(rows):
    values=costs(rows)
    return sorted(i for out in range(3) for i in sorted(
        (j for j,r in enumerate(CATALOG) if r['output']==out),key=lambda j:(values[j],j))[:3])


def learner_digest(rows,tick):
    return sha256(encode({'catalog':CATALOG,'tick':tick,'rows':rows})).hexdigest()


def validate_snapshot(snapshot,expected_point):
    point(expected_point)
    if type(snapshot) is not dict or set(snapshot)!={'schema','point','provenance','pool','rows','proposed_pool','learner_sha256'}:
        raise ValueError('Missing or extra snapshot fields')
    if encode(snapshot['point'])!=encode(expected_point) or snapshot['schema']!='sara-route-evidence-snapshot-v1':raise ValueError('Boundary/schema mismatch')
    if snapshot['provenance'] not in ('symbolic_contract_fixture','future_external_state'):
        raise ValueError('Explicit snapshot provenance required')
    validate_rows(snapshot['rows'],expected_point['tick'])
    active=pool(snapshot['pool'])
    expected_size=45 if expected_point['stage']=='pre_initial' else 9
    if len(active)!=expected_size:raise ValueError('Wrong active pool at boundary')
    if snapshot['proposed_pool']!=ranked_pool(snapshot['rows']):raise ValueError('Ranking differs from unchanged local rule')
    pool(snapshot['proposed_pool'],9)
    if snapshot['learner_sha256']!=learner_digest(snapshot['rows'],expected_point['tick']):raise ValueError('Learner-field digest differs')
    if len(encode(snapshot))>LIMITS['bytes_per_snapshot']:raise ValueError('Per-snapshot byte ceiling exceeded')
    return snapshot


def capture(state,boundary,provenance):
    """Copy supplied existing fields; no learning, prediction, labels or mutation.

    This validates asserted state, not its execution provenance or causal truth.
    A future executor must bind the source and invoke only authorized hooks.
    """
    point(boundary)
    if type(state) is not dict or set(state)!=STATE_FIELDS:raise ValueError('Explicit learner-field view required')
    if state['pending'] is not None or integer(state['steps'])!=boundary['tick']:raise ValueError('Pending-free matching causal boundary required')
    routes=state['routes']
    if type(routes) is not tuple or len(routes)!=45:raise ValueError('Candidate catalog changed')
    supplied=[]
    for route in routes:
        if type(route) is not tuple or len(route)!=2 or type(route[0]) is not tuple or any(type(i) is not int for i in route[0]) or type(route[1]) is not int:raise ValueError('Invalid route catalog primitive types')
        supplied.append({'inputs':list(route[0]),'output':route[1]})
    if encode(supplied)!=encode(CATALOG):raise ValueError('Candidate catalog changed')
    for field in ('tables','errors','weights','observations'):
        if type(state[field]) is not list or len(state[field])!=45:raise ValueError('Complete candidate state required')
    rows=[]
    for i,route in enumerate(CATALOG):
        table=state['tables'][i]
        if type(table) is not dict or len(table)>2**len(route['inputs']):raise ValueError('Invalid local key table')
        keys=list(product((0,1),repeat=len(route['inputs'])))
        for key,counts in table.items():
            if type(key) is not tuple or len(key)!=len(route['inputs']) or any(type(bit) is not int or bit not in (0,1) for bit in key):raise ValueError('Invalid causal binary address')
            if key not in keys or type(counts) is not list or len(counts)!=2:raise ValueError('Invalid class-count cell')
            for n in counts:integer(n)
        vector=[n for key in keys for n in table.get(key,[0,0])]
        rows.append([state['observations'][i],state['errors'][i],state['weights'][i],vector])
    validate_rows(rows,state['steps'])
    active=list(state['pool']) if type(state['pool']) in (tuple,list) else None
    result={'schema':'sara-route-evidence-snapshot-v1','point':deepcopy(boundary),'provenance':provenance,
            'pool':active,'rows':rows,'proposed_pool':ranked_pool(rows),
            'learner_sha256':learner_digest(rows,state['steps'])}
    # Validate before releasing any record; a failure emits no clipped snapshot.
    return validate_snapshot(result,boundary)


def compare_pair(before,after):
    a,b=before['point'],after['point']
    validate_snapshot(before,a);validate_snapshot(after,b)
    if {k:v for k,v in a.items() if k!='stage'}!={k:v for k,v in b.items() if k!='stage'}:
        raise ValueError('Pair crosses case/arm/tick')
    if not a['stage'].startswith('pre_') or b['stage']!='post_'+a['stage'][4:]:raise ValueError('Wrong pre/post boundary pair')
    if before['provenance']!=after['provenance']:raise ValueError('Pair crosses provenance classes')
    if before['learner_sha256']!=after['learner_sha256'] or encode(before['rows'])!=encode(after['rows']):raise ValueError('Selection hook changed learner fields')
    if a['stage']=='pre_start':
        if before['pool']!=after['pool']:raise ValueError('Exploration start changed readout pool')
    elif after['pool']!=after['proposed_pool']:raise ValueError('Commit/initial pool does not match fixed ranking')
    return True


def diagnose(snapshot,true_route_ids=None):
    """Separate evaluator-only assertions; never pass these back to a learner."""
    validate_snapshot(snapshot,snapshot['point'])
    if true_route_ids is None:
        return {'evaluator_ids_available':False,'dependencies':None,'causal_attribution':None}
    ids=pool(true_route_ids,3)
    if sorted(CATALOG[i]['output'] for i in ids)!=[0,1,2]:raise ValueError('One evaluator route per output required')
    values=costs(snapshot['rows']);results=[]
    for i in ids:
        group=sorted((j for j,r in enumerate(CATALOG) if r['output']==CATALOG[i]['output']),key=lambda j:(values[j],j))
        rank=group.index(i)+1;cutoff=values[group[2]]
        results.append({'route_id':i,'observations':snapshot['rows'][i][0],
            'support_ready':snapshot['rows'][i][0]>=16,'retained':i in snapshot['pool'],
            'rank':rank,'cost_margin_to_third':values[i]-cutoff,
            'rank_exclusion_kind':None if rank<=3 else ('tie_break' if values[i]==cutoff else 'score'),
            'support_is_causal_failure':None,'functional_substitution_proven':None})
    return {'evaluator_ids_available':True,'dependencies':results,'causal_attribution':None}


def window_support(start,end):
    """Derive recent support from captured endpoints; never reset/retrain state."""
    a,b=start['point'],end['point']
    validate_snapshot(start,a);validate_snapshot(end,b)
    if a['stage']!='pre_start' or b['stage']!='pre_commit' or b['tick']-a['tick']!=180 or a['case_id']!=b['case_id'] or a['arm']!=b['arm'] or start['provenance']!=end['provenance'] or start['pool']!=end['pool']:
        raise ValueError('Matching original exploration endpoints required')
    rows=[]
    for old,new in zip(start['rows'],end['rows']):
        count=new[0]-old[0];error=new[1]-old[1];weight=new[2]-old[2]
        vector=[n-o for o,n in zip(old[3],new[3])]
        if count!=36 or any(n<0 for n in vector) or sum(vector)!=count or not 0.0<error<weight:
            raise ValueError('Window evidence lost/reset or balanced exploration changed')
        rows.append([count,error,weight,vector])
    if sum(r[2] for r in rows)!=global_weight(b['tick'])-global_weight(a['tick']):raise ValueError('Window public-time weight conservation failed')
    return {'route_window_rows':rows,'window_ranked_pool':ranked_pool(rows),
            'current_cumulative_ranked_pool':end['proposed_pool'],
            'recent_support_is_causal_failure':None,
            'counterfactual_reset_or_windowed_learning_accuracy':None}


def encode_collection(header,snapshots,roster):
    if type(header) is not dict or type(snapshots) is not list:raise ValueError('Explicit collection objects required')
    if type(roster) is not list or not 1<=len(roster)<=160 or len(snapshots)!=len(roster):raise ValueError('Missing/extra registered snapshot')
    if header.get('roster_sha256')!=sha256(encode(roster)).hexdigest():raise ValueError('Collection not bound to its declared roster')
    if set(header)!={'schema','contract_sha256','catalog','roster_sha256','provenance','executor_binding'} or header['schema']!='sara-route-evidence-collection-v1' or header['contract_sha256']!=PROTOCOL_SHA256 or encode(header['catalog'])!=encode(CATALOG):raise ValueError('Collection schema/catalog/contract mismatch')
    if header['provenance']=='symbolic_contract_fixture':
        if header['executor_binding'] is not None:raise ValueError('Symbolic fixture must not claim execution')
    elif header['provenance']=='future_external_state':
        binding=header['executor_binding']
        if type(binding) is not dict or set(binding)!={'registration_sha256','execution_sha256'} or any(type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in binding.values()):raise ValueError('Future collection needs separate executor bindings')
    else:raise ValueError('Explicit collection provenance required')
    if any(s['provenance']!=header['provenance'] for s in snapshots):raise ValueError('Mixed fixture/execution collection')
    chunks=[encode(validate_snapshot(s,p)) for s,p in zip(snapshots,roster)]
    if len({encode(p) for p in roster})!=len(roster):raise ValueError('Duplicate planned snapshot')
    index=0
    while index<len(snapshots):
        if roster[index]['stage']=='final':
            index+=1
        else:
            if index+1>=len(snapshots):raise ValueError('Missing paired boundary snapshot')
            compare_pair(snapshots[index],snapshots[index+1])
            index+=2
    prefix=b'{"header":'+encode(header)+b',"snapshots":[';suffix=b']}'
    if len(prefix)+len(suffix)>65536:raise ValueError('Header/framing byte ceiling exceeded')
    raw=prefix+b','.join(chunks)+suffix
    if len(raw)>2097152:raise ValueError('Complete report bound exceeded')
    return raw


def symbolic_state():
    """Consistent artificial metadata, not an old or newly executed model state."""
    return {'routes':tuple((tuple(r['inputs']),r['output']) for r in CATALOG),
            'tables':[{tuple(0 for _ in r['inputs']):[36,0]} for r in CATALOG],
            'errors':[20.0]*45,'weights':[80.0]*45,'observations':[36]*45,
            'steps':180,'pending':None,'pool':tuple(range(45))}
