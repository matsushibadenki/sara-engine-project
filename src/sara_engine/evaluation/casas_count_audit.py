"""Read-only support and effect decomposition of fixed fitted count tables."""
import ast
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from sara_engine.evaluation.casas_evidence import evidence
from sara_engine.evaluation.sparse_count_replay import SparseCapacityCounts

CATEGORIES = ('no_complement', 'support_impossible', 'margin_impossible',
              'observed_below_bound', 'evidence_pass')
MAX_EXPANDED = 32*1024*1024


def classify(detail, current):
    """Separate support, fixed outcome-margin and observed-separation constraints."""
    if len(detail) != 2 or len(current) != 2 or any(type(v) is not int or v < 0 for v in (*detail,*current)):
        raise ValueError('Nonnegative integer binary counts required')
    complement = tuple(c-d for c,d in zip(current,detail))
    if min(complement) < 0 or not sum(detail):
        raise ValueError('Observed detail must be a nonempty subset of current')
    n, m = sum(detail), sum(complement)
    if not m:
        return {'category':'no_complement','detail_observations':n,
                'complement_observations':0,'radius':None,'observed_separation':None,
                'margin_maximum_separation':None,'evidence':False}
    radius = math.sqrt(math.log(80)/(2*n))+math.sqrt(math.log(80)/(2*m))
    separation = abs(detail[1]/n-complement[1]/m)
    # With fixed n, m and total positives, the absolute affine contrast is
    # maximized at an endpoint of feasible integer detail-positive allocations.
    positives = current[1]
    lower, upper = max(0,positives-m), min(n,positives)
    maximum = max(abs(x/n-(positives-x)/m) for x in (lower,upper))
    if radius >= 1:
        category = 'support_impossible'
    elif maximum <= radius:
        category = 'margin_impossible'
    elif separation <= radius:
        category = 'observed_below_bound'
    else:
        category = 'evidence_pass'
    passed = evidence(detail,current)
    if passed != (category == 'evidence_pass'):
        raise ValueError('Audit differs from unchanged evidence rule')
    return {'category':category,'detail_observations':n,'complement_observations':m,
            'radius':radius,'observed_separation':separation,
            'margin_maximum_separation':maximum,'evidence':passed}


def parse_key(key):
    if not isinstance(key,str) or len(key) > 4096:
        raise ValueError('Invalid or oversized address')
    try:
        value = ast.literal_eval(key)
    except (SyntaxError,ValueError,RecursionError) as error:
        raise ValueError('Malformed address') from error
    if not isinstance(value,tuple) or not value:
        raise ValueError('Tuple address required')
    return value


def load_snapshot(path, arm):
    """Bound decompression and validate the frozen count schema without learning."""
    with gzip.open(Path(path),'rb') as stream:
        raw = stream.read(MAX_EXPANDED+1)
    if len(raw) > MAX_EXPANDED:
        raise ValueError('Expanded checkpoint exceeds audit bound')
    state = json.loads(raw)
    if state.get('policy') != 'same_sensor_fast10_four_branches_v1' or state.get('arm') != arm:
        raise ValueError('Unexpected frozen model identity')
    SparseCapacityCounts.restore(state['counts'])
    return state


def audit(state):
    """Preserve exact current keys; verify detail partitions before summarizing."""
    arm = state['arm']
    if arm not in ('pool_evidence','joint_evidence'):
        raise ValueError('Unregistered audit arm')
    model = SparseCapacityCounts.restore(state['counts'])
    cells = model.cells
    currents, details = {}, defaultdict(dict)
    for key, counts in cells.items():
        if key == 'global':
            continue
        parsed = parse_key(key)
        if parsed[0] == 'current' and len(parsed) == 2:
            context = parsed[1]
            if (not isinstance(context,tuple) or len(context) != 4
                    or not isinstance(context[0],tuple) or len(context[0]) != 2
                    or any(not isinstance(v,str) for v in context[0])
                    or any(type(v) is not int for v in context[1:])
                    or not (0 <= context[1] < 7 and 0 <= context[2] < 24 and 0 <= context[3] < 4)):
                raise ValueError('Invalid exact current context')
            currents[key] = counts
        elif arm == 'pool_evidence' and parsed[0] == 'pool' and len(parsed) == 4:
            if type(parsed[2]) is not bool or type(parsed[3]) is not bool:
                raise ValueError('Invalid pooled branch features')
            details[parsed[1]][key] = counts
        elif arm == 'joint_evidence' and parsed[0] == 'detail' and len(parsed) == 2 and isinstance(parsed[1],tuple) and len(parsed[1]) == 4:
            current_key = repr(('current',parsed[1][0]))
            details[current_key][key] = counts
        else:
            raise ValueError('Unexpected address structure')
    if set(currents) != set(details) or 'global' not in cells:
        raise ValueError('Current/detail address partition differs')
    totals = [sum(c[label] for c in currents.values()) for label in (0,1)]
    if cells['global'] != totals or model.changed_writes != 3*sum(totals):
        raise ValueError('Global counts or three-write conservation differs')
    categories = Counter({name:0 for name in CATEGORIES})
    weighted = Counter({name:0 for name in CATEGORIES})
    context_categories = {name:set() for name in CATEGORIES}
    records, opposing = [], 0
    for current_key, branches in sorted(details.items()):
        current = currents[current_key]
        if [sum(c[label] for c in branches.values()) for label in (0,1)] != current:
            raise ValueError('Details do not partition their exact current context')
        if arm == 'pool_evidence' and len(branches) > 4:
            raise ValueError('Four-branch bound exceeded')
        signs = set()
        for key, detail in sorted(branches.items()):
            item = classify(detail,current)
            category = item['category']
            categories[category] += 1
            weighted[category] += sum(detail)
            context_categories[category].add(current_key)
            if item['evidence']:
                sign = 1 if detail[1]/sum(detail) > current[1]/sum(current) else -1
                if detail[1 if sign > 0 else 0] >= 4:
                    signs.add(sign)
            records.append({'current_key':current_key,'detail_key':key,**item})
        opposing += signs == {-1,1}
    if sum(categories.values()) != len(records) or sum(weighted.values()) != sum(totals):
        raise ValueError('Audit conservation failed')
    return {'arm':arm,'fit_observations':sum(totals),'existing_changed_writes':model.changed_writes,
            'audit_learning_writes':0,
            'current_counts_sha256':sha256(json.dumps(currents,sort_keys=True).encode()).hexdigest(),
            'current_contexts':len(currents),'detail_branches':len(records),
            'categories':dict(categories),'fit_observations_by_category':dict(weighted),
            'contexts_by_category':{k:len(v) for k,v in context_categories.items()},
            'opposing_evidence_contexts':opposing,'branches':records}
