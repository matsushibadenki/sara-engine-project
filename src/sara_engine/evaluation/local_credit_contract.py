"""Design-only local-credit proposal checks; never apply learning updates."""
from copy import deepcopy

ARMS = ('current', 'history', 'eligibility', 'disabled', 'permuted', 'scalar_emulator')
LIMITS = {'arms': 6, 'history_events': 8, 'writes_per_arm': 16,
          'state_slots': 4096, 'value_bound': 1000000, 'artifact_bytes': 65536}


def fields(value, keys):
    if type(value) is not dict or set(value) != set(keys.split()):
        raise ValueError('Missing or unexpected contract fields')


def number(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Expected bounded integer')
    return value


def identifier(value):
    if type(value) is not str or not 1 <= len(value) <= 64 or not value.isascii():
        raise ValueError('Expected bounded ASCII identifier')
    return value


def digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected observer SHA-256')
    return value


def validate(proposal):
    """Check asserted metadata, not truth of execution or a learning equation.

    No proposal is applied or consumed. Receipt reuse checks are relative to the
    supplied bounded consumed-ID set, which must be proved by a future executor.
    """
    fields(proposal, 'receipt consumed events arms')
    receipt = proposal['receipt']
    fields(receipt, 'id created outcome delivered expires status observer')
    rid = identifier(receipt['id'])
    created = number(receipt['created'], 0, 10**12)
    outcome = number(receipt['outcome'], created+300, created+600)
    delivered = number(receipt['delivered'], outcome, created+600)
    number(receipt['expires'], delivered, created+600)
    if receipt['status'] != 'delivered':
        raise ValueError('Only a delivered result may propose credit')
    observer = digest(receipt['observer'])
    consumed = proposal['consumed']
    if type(consumed) is not list or len(consumed) > 64:
        raise ValueError('Consumed-ID set exceeds static design bound')
    if len({identifier(i) for i in consumed}) != len(consumed) or rid in consumed:
        raise ValueError('Duplicate or already consumed receipt')
    events = proposal['events']
    if type(events) is not dict or not 1 <= len(events) <= LIMITS['history_events']:
        raise ValueError('Shared history is outside original bound')
    for event, time in events.items():
        identifier(event)
        number(time, created-300, created)
    arms = proposal['arms']
    if type(arms) is not dict or set(arms) != set(ARMS):
        raise ValueError('All sufficient controls are required')
    counts = {}
    for name in ARMS:
        arm = arms[name]
        fields(arm, 'observer state_slots routes updates')
        if digest(arm['observer']) != observer:
            raise ValueError('Shared observation differs')
        number(arm['state_slots'], 1, LIMITS['state_slots'])
        routes = arm['routes']
        if type(routes) is not dict or not routes or len(routes) > LIMITS['state_slots']//2:
            raise ValueError('Invalid bounded route state')
        for route, state in routes.items():
            identifier(route)
            fields(state, 'weight observations events')
            number(state['weight'], -LIMITS['value_bound'], LIMITS['value_bound'])
            number(state['observations'], 0, LIMITS['value_bound']-1)
            if type(state['events']) is not list or not state['events'] or len(state['events']) > len(events):
                raise ValueError('Local causal membership required')
            for event in state['events']:
                identifier(event)
            if len(set(state['events'])) != len(state['events']) or any(e not in events for e in state['events']):
                raise ValueError('Private or duplicated causal event')
        if arm['state_slots'] < 2*len(routes):
            raise ValueError('Declared slots omit route values')
        updates = arm['updates']
        if type(updates) is not list or not 1 <= len(updates) <= LIMITS['writes_per_arm']//2:
            raise ValueError('Positive bounded changed-field proposal required')
        seen = set()
        for update in updates:
            fields(update, 'receipt route event old_weight new_weight old_observations new_observations')
            route = identifier(update['route'])
            if update['receipt'] != rid or route not in routes or route in seen:
                raise ValueError('Foreign receipt, unknown route or duplicate update')
            seen.add(route)
            state = routes[route]
            if identifier(update['event']) not in state['events']:
                raise ValueError('Update is not locally causally eligible')
            for key in ('old_weight', 'new_weight'):
                number(update[key], -LIMITS['value_bound'], LIMITS['value_bound'])
            for key in ('old_observations', 'new_observations'):
                number(update[key], 0, LIMITS['value_bound'])
            if update['old_weight'] != state['weight'] or update['old_observations'] != state['observations']:
                raise ValueError('Stale previous state')
            if update['new_weight'] == update['old_weight'] or update['new_observations'] != update['old_observations']+1:
                raise ValueError('No-op or padded counter update')
        counts[name] = 2*len(updates)
    if len(set(counts.values())) != 1 or len({a['state_slots'] for a in arms.values()}) != 1:
        raise ValueError('Changed-field counts or declared state ceilings differ')
    if arms['eligibility'] != arms['scalar_emulator']:
        raise ValueError('Equation-matched emulator must have identical proposals/state')
    return {'structural_contract_passed': True, 'proposed_changed_fields': counts,
            'actual_applied_writes': 0, 'experiment_authorized': False,
            'resource_measurement_authorized': False,
            'independent_task_qualified': None, 'causal_effect_established': None,
            'scope': 'Symbolic metadata only; no execution, learning-rule proof or measured write parity'}


def witness():
    """A symbolic two-route proposal; no prediction examples or outcomes scored."""
    observer = 'a'*64
    routes = {r: {'weight': 0, 'observations': 0, 'events': [e]}
              for r, e in (('left', 'e0'), ('right', 'e1'))}
    updates = [{'receipt': 'r0', 'route': r, 'event': e, 'old_weight': 0,
                'new_weight': 1, 'old_observations': 0, 'new_observations': 1}
               for r, e in (('left', 'e0'), ('right', 'e1'))]
    arm = {'observer': observer, 'state_slots': 16, 'routes': routes, 'updates': updates}
    return {'receipt': {'id': 'r0', 'created': 300, 'outcome': 600, 'delivered': 900,
                        'expires': 900, 'status': 'delivered', 'observer': observer},
            'consumed': [], 'events': {'e0': 0, 'e1': 300},
            'arms': {name: deepcopy(arm) for name in ARMS}}
