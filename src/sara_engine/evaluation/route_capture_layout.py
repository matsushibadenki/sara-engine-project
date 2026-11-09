"""Read-only logical projection of typed capture records; no model calculations."""
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations,product
import json
from math import isfinite
from sara_engine.evaluation.route_capture_candidate import CapturePair,PhaseCapture
from sara_engine.evaluation.route_outcome_admission import Input

ROUTES=tuple((key,out) for width in (1,2) for key in combinations(range(5),width) for out in range(3))


class LayoutError(ValueError):
    """Malformed or inconsistent capture layout, never an authorization."""


def require(value,message):
    if not value:raise LayoutError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def decode(encoded,*,depth_limit=24,node_limit=20000):
    remaining=node_limit
    def visit(node,depth):
        nonlocal remaining
        remaining-=1
        require(remaining>=0 and depth<=depth_limit,'Typed state bound exceeded')
        require(type(node) is tuple and len(node) in (1,2) and type(node[0]) is str,'Invalid typed node')
        tag=node[0]
        if tag=='none':
            require(len(node)==1,'Invalid None node');return None
        require(len(node)==2,'Missing typed payload');value=node[1]
        if tag=='bool':require(type(value) is bool,'Invalid Boolean');return value
        if tag=='int':
            require(type(value) is str and 0<len(value)<=24,'Invalid integer encoding')
            try:result=int(value)
            except ValueError:raise LayoutError('Invalid integer encoding') from None
            require(str(result)==value,'Noncanonical integer');return result
        if tag=='float':
            require(type(value) is str and len(value)<=32,'Invalid float encoding')
            try:result=float.fromhex(value)
            except ValueError:raise LayoutError('Invalid float encoding') from None
            require(isfinite(result) and result.hex()==value,'Nonfinite/noncanonical float');return result
        if tag=='str':require(type(value) is str,'Invalid string');return value
        require(tag in ('list','tuple','dict') and type(value) is tuple,'Unknown typed tag/container')
        require(len(value)<=node_limit,'Container bound exceeded')
        if tag in ('list','tuple'):
            result=[visit(n,depth+1) for n in value]
            return result if tag=='list' else tuple(result)
        result={};order=[]
        for pair in value:
            require(type(pair) is tuple and len(pair)==2,'Invalid encoded dict pair')
            key=visit(pair[0],depth+1);item=visit(pair[1],depth+1)
            try:
                require(key not in result,'Duplicate encoded key');result[key]=item
            except TypeError:raise LayoutError('Unhashable encoded key') from None
            order.append(canonical(pair[0]))
        require(order==sorted(order),'Noncanonical dictionary ordering')
        return result
    return visit(encoded,0)


@dataclass(frozen=True,slots=True)
class Projection:
    fields:tuple
    allocations:tuple


def project(state):
    require(type(state) is dict,'Expected model dictionary')
    required={'routes','tables','observations','errors','weights','pending','clock','steps','updates'}
    require(required<=set(state),'Missing logical state field')
    require(type(state['routes']) is tuple and state['routes']==ROUTES
            and all(type(route) is tuple and len(route)==2 and type(route[0]) is tuple
                    and all(type(i) is int for i in route[0]) and type(route[1]) is int for route in state['routes']), 'Route catalog differs')
    for field in ('tables','observations','errors','weights'):
        require(type(state[field]) is list and len(state[field])==45,'Invalid per-route array: '+field)
    cells=[];allocations=[]
    for r,(inputs,_) in enumerate(ROUTES):
        table=state['tables'][r]
        require(type(table) is dict,'Invalid route table')
        for key,masses in table.items():
            require(type(key) is tuple and len(key)==len(inputs)
                    and all(type(bit) is int and bit in (0,1) for bit in key),'Nonbinary route key')
            require(type(masses) is list and len(masses)==2
                    and all(type(m) is int and m>=0 for m in masses),'Invalid class masses')
        for rank,key in enumerate(product((0,1),repeat=len(inputs))):
            allocated=key in table;allocations.append((r,rank,allocated))
            masses=table[key] if allocated else (0,0)
            cells.extend((((r,'mass_left',rank),masses[0]),((r,'mass_right',rank),masses[1])))
        obs,error,weight=(state[n][r] for n in ('observations','errors','weights'))
        require(type(obs) is int and obs>=0,'Invalid observation count')
        require(type(error) is float and type(weight) is float and isfinite(error) and isfinite(weight)
                and 0<=error<=weight,'Invalid error/weight fields')
        cells.extend((((r,'observations',-1),obs),((r,'error_sum',-1),error),((r,'weight_sum',-1),weight)))
    for name in ('clock','steps','updates'):
        require(type(state[name]) is int and state[name]>=(-1 if name=='clock' else 0),'Invalid lifecycle counter')
    return Projection(tuple(sorted(cells)),tuple(allocations))


def validate(pair,*,byte_limit=262144):
    require(type(pair) is CapturePair and type(pair.input) is Input
            and type(pair.predecessor) is PhaseCapture and type(pair.receipt) is PhaseCapture,'Invalid capture record types')
    require(len(canonical((pair.predecessor.state,pair.receipt.state)))<=byte_limit,'Encoded pair byte bound exceeded')
    states=[];projections=[]
    for phase in (pair.predecessor,pair.receipt):
        require((phase.owner_label,phase.epoch_label)==(pair.input.owner,pair.input.epoch),'Capture owner/epoch differs')
        require(type(phase.revision) is int and phase.revision>=0 and phase.revision%2==0,'Invalid stable revision')
        require(type(phase.digest) is str and sha256(canonical(phase.state)).hexdigest()==phase.digest,'Capture digest differs')
        state=decode(phase.state);states.append(state);projections.append(project(state))
    require(pair.receipt.revision==pair.predecessor.revision+2,'Paired revisions differ')
    before,after=states
    require(before['pending'] is None,'Predecessor already has pending receipt')
    require(before['steps']==after['steps']==pair.input.receiving_id and before['updates']==after['updates'],'Begin changed learning counters')
    require(before['clock']<pair.input.receiving_id and after['clock']==pair.input.receiving_id,'Receipt clock differs')
    require(canonical((projections[0].fields,projections[0].allocations))==canonical((projections[1].fields,projections[1].allocations)), 'Begin changed learning fields or allocation')
    pending=after['pending']
    require(type(pending) is tuple and len(pending)==5,'Invalid already-recorded pending receipt')
    tick,events,active,probabilities,prediction=pending
    require(type(tick) is int and tick==pair.input.receiving_id and type(events) is tuple
            and events==pair.input.events and all(type(v) is int for v in events),'Pending identity/events differ')
    require(type(active) is tuple and len(active)==len(set(active))==9
            and all(type(r) is int and 0<=r<45 for r in active),'Incomplete/duplicate active routes')
    require(all(sum(ROUTES[r][1]==slot for r in active)==3 for slot in range(3)),'Unbalanced active routes')
    require(type(probabilities) is tuple and len(probabilities)==9,'Invalid receipt probabilities')
    require(type(prediction) is tuple and len(prediction)==3,'Invalid recorded prediction')
    def probability_type(p):return p is None or type(p) is float and isfinite(p) and 0<=p<=1
    require(all(probability_type(p) for p in prediction),'Invalid recorded prediction type')
    credits=[]
    for route,p in zip(active,probabilities):
        key=tuple(events[i] for i in ROUTES[route][0]);allocated=key in after['tables'][route]
        require(probability_type(p) and ((p is not None)==allocated),'Probability/allocation mismatch')
        credits.append((route,key,allocated,p))
    require(len(projections[0].fields)==435 and len(projections[0].allocations)==150,'Logical catalog size differs')
    return {'fields':projections[0].fields,'allocations':projections[0].allocations,'receipt_credits':tuple(credits),
            'learning_fields':435,'allocation_flags':150,'begin_learning_changes':0,
            'probabilities_recomputed':False,'authority_authenticated':False,'capture_unblocked':False,'application_authorized':False}
