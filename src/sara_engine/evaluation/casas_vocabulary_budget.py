"""Address-space certificates from frozen event vocabularies, without learning."""
from hashlib import sha256
import json
from sara_engine.evaluation.casas_count_audit import audit, parse_key
from sara_engine.evaluation.casas_observation_budget import ADDITIONAL_OBSERVATIONS, STATE_CAPACITY

CLOCK_CONTEXTS = 7*24*4
GAP_PAIRS = 4*4
ARMS = ('pool_evidence','joint_evidence')


def universe_bound(events, arm):
    """Conservative syntactic route ceiling, including shared global/current cells."""
    if type(events) is not int or events < 1 or arm not in ARMS:
        raise ValueError('Positive event cardinality and registered arm required')
    current = CLOCK_CONTEXTS*events
    detail = 4*current if arm == 'pool_evidence' else GAP_PAIRS*current*events*events
    return 1+current+detail


def certified_vocabulary_limit(arm):
    """Largest cardinality whose complete syntactic envelope fits the fixed cap."""
    low,high = 0,STATE_CAPACITY//CLOCK_CONTEXTS
    while low < high:
        middle = (low+high+1)//2
        if universe_bound(middle,arm) <= STATE_CAPACITY:
            low = middle
        else:
            high = middle-1
    return low


def event(value):
    if (not isinstance(value,tuple) or len(value) != 2
            or any(not isinstance(v,str) or not v for v in value)
            or value[1] not in ('ON','OFF','OPEN','CLOSE','CLOSED')):
        raise ValueError('Saved binary event schema differs')
    return value


def vocabulary(state):
    """Recover only events retained in current and joint-detail address keys."""
    validated = audit(state)
    currents,history = set(),set()
    for key in state['counts']['cells']:
        if key == 'global':
            continue
        address = parse_key(key)
        if address[0] == 'current':
            currents.add(event(address[1][0]))
        elif state['arm'] == 'joint_evidence':
            history.update(event(v) for v in address[1][1:3])
            gaps = address[1][3]
            if (not isinstance(gaps,tuple) or len(gaps) != 2
                    or any(type(v) is not int or not 0 <= v < 4 for v in gaps)):
                raise ValueError('Saved gap-bin schema differs')
    return currents,history,validated


def analyze(states):
    if set(states) != set(ARMS):
        raise ValueError('Both preregistered fitted states required')
    summaries,all_current,all_history = {},set(),set()
    for arm,state in states.items():
        if state['arm'] != arm or state['counts']['capacity'] != STATE_CAPACITY:
            raise ValueError('Checkpoint identity or capacity differs')
        current,history,summary = vocabulary(state)
        summaries[arm] = summary
        all_current.update(current)
        all_history.update(history)
    if len({(s['current_counts_sha256'],s['existing_changed_writes']) for s in summaries.values()}) != 1:
        raise ValueError('Current conditions or historical write matching differ')
    known = all_current | all_history
    if not known:
        raise ValueError('Empty vocabulary')
    inventory = {'current_events':sorted(all_current),'stored_history_events':sorted(all_history),
                 'shared_known_events':sorted(known),'history_only_events':sorted(all_history-all_current)}
    report = {'current_event_cardinality':len(all_current),'stored_history_event_cardinality':len(all_history),
              'shared_known_event_cardinality':len(known),'history_only_event_cardinality':len(all_history-all_current),
              'clock_contexts_per_event':CLOCK_CONTEXTS,'gap_pair_cardinality':GAP_PAIRS,
              'known_vocabulary_sha256':sha256(json.dumps(sorted(known)).encode()).hexdigest(),
              'state_capacity':STATE_CAPACITY,'additional_qualified_outcome_ceiling':ADDITIONAL_OBSERVATIONS,
              'actual_acquired_observations':0,'actual_learning_writes':0,
              'raw_source_opened':False,'confirmation_opened':False,
              'original_development_opened':False,'test_opened':False,'arms':{},'inventory':inventory}
    for arm,state in sorted(states.items()):
        routes = len(state['counts']['cells'])
        bound = universe_bound(len(known),arm)
        if routes > bound:
            raise ValueError('Existing routes exceed proven address envelope')
        limit = certified_vocabulary_limit(arm)
        full_budget_generic = routes+2*ADDITIONAL_OBSERVATIONS
        at_next = universe_bound(len(known)+1,arm)
        report['arms'][arm] = {
            'current_counts_sha256':summaries[arm]['current_counts_sha256'],
            'existing_changed_writes':summaries[arm]['existing_changed_writes'],
            'existing_routes':routes,
            'exact_address_recurrence_bound':routes,
            'known_event_new_combination_bound':bound,
            'known_event_bound_at_acquisition_ceiling':min(bound,full_budget_generic),
            'known_event_capacity_certified':bound <= STATE_CAPACITY,
            'capacity_certified_total_event_limit':limit,
            'additional_unknown_event_types_certified':limit-len(known) if len(known) <= limit else None,
            'one_unknown_event_type_bound':at_next,
            'one_unknown_event_type_capacity_certified':at_next <= STATE_CAPACITY,
            'unrestricted_event_bound_at_acquisition_ceiling':full_budget_generic,
            'unrestricted_event_capacity_certified':full_budget_generic <= STATE_CAPACITY,
            'unlimited_horizon_unrestricted_event_bound':None,
            'unrestricted_feasibility_status':'Unresolved; finite acquisition upper bound is not a measured growth curve',
            'new_learning_writes':0}
    return report
