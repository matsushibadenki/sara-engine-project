"""Independent scalar transitions on fixed, nonlearning hand snapshots only."""
from itertools import combinations
from math import copysign, isfinite

# Catalog construction is independent of all production/fixture helpers.
WIDTHS = tuple(width for width in (1, 2) for _ in combinations(range(5), width) for _ in range(3))
CATALOG = tuple(sorted([(r, side, key) for r, width in enumerate(WIDTHS)
                        for key in range(2 ** width) for side in ('mass_left', 'mass_right')]
                       + [(r, field, -1) for r in range(45)
                          for field in ('observations', 'error_sum', 'weight_sum')]))
LIMIT = 6717440


def demand(condition, message):
    if not condition: raise ValueError(message)


def stamp(address, value):
    """Strict typed value without coercion or numeric tolerance."""
    if address[1] in ('mass_left', 'mass_right', 'observations'):
        bound = 205 if address[1] == 'observations' else LIMIT
        demand(type(value) is int and 0 <= value <= bound, 'Invalid exact integer state')
        return ('int', str(value))
    demand(type(value) is float and isfinite(value) and 0 <= value <= LIMIT
           and not (value == 0 and copysign(1, value) < 0), 'Invalid exact float state')
    return ('float', value.hex())


def validate_fields(fields):
    demand(type(fields) is tuple and len(fields) == 435, 'Invalid immutable field catalog')
    addresses = []
    for cell in fields:
        demand(type(cell) is tuple and len(cell) == 2, 'Invalid immutable field cell')
        address, value = cell
        demand(type(address) is tuple and len(address) == 3
               and type(address[0]) is int and type(address[1]) is str
               and type(address[2]) is int, 'Invalid address types')
        stamp(address, value)
        addresses.append(address)
    demand(tuple(addresses) == CATALOG, 'Missing/duplicate/foreign field')
    values = dict(fields)
    for route in range(45):
        demand(values[(route, 'error_sum', -1)] <= values[(route, 'weight_sum', -1)], 'Error exceeds normalizer')


def transition(owner, before, pending, delivery, diagnostic, consumed):
    """Return a fresh fixture state; no expected after-values enter this API."""
    demand(type(owner) is tuple and len(owner) == 6, 'Invalid owner')
    seed, task, condition, history, error_rule, role = owner
    demand(type(seed) is int and seed == 1201 and task == 'independent' and condition == 'changed'
           and type(history) is str and history in ('cumulative', 'discounted')
           and type(error_rule) is str and error_rule in ('uniform', 'public_exponential')
           and type(role) is str and role in ('candidate', 'permanent', 'shuffled_outcome'), 'Unknown fixture owner')
    validate_fields(before)
    demand(type(diagnostic) is tuple and len(diagnostic) == 2 and diagnostic[0] == 'diagnostic'
           and type(diagnostic[1]) is tuple and len(diagnostic[1]) == 9
           and len(set(diagnostic[1])) == 9
           and all(type(r) is int and 0 <= r < 45 for r in diagnostic[1]), 'Diagnostic/training namespace violation')
    demand(type(consumed) is frozenset and all(type(item) is tuple and len(item) == 2
           and item[0] == owner and type(item[1]) is int and 0 <= item[1] < 1024 for item in consumed), 'Invalid immutable consumed ledger')
    demand(type(pending) is tuple and len(pending) == 4 and pending[0] == owner, 'Foreign pending receipt')
    _, tick, source, credits = pending
    demand(type(tick) is int and tick in (0, 64, 1023) and type(source) is int
           and 0 <= source < 1024 and source // 512 == tick // 512, 'Invalid receipt identity/phase')
    demand(role == 'shuffled_outcome' or source == tick, 'Nonshuffled source differs')
    demand((owner, tick) not in consumed, 'Duplicate consumed receipt')
    expected = []
    for slot in range(3):
        group = tuple(range(slot, 45, 3))
        for j in range(3):
            route = group[(3 * tick + j) % 15]
            expected.append((route, tick % (2 ** WIDTHS[route]),
                             'mass_left' if len(expected) % 2 == 0 else 'mass_right'))
    demand(type(credits) is tuple and all(type(c) is tuple and len(c) == 3
           and type(c[0]) is int and type(c[1]) is int and type(c[2]) is str for c in credits)
           and credits == tuple(expected), 'Pinned route/key/side drift')
    demand(type(delivery) is tuple and len(delivery) == 5 and delivery[0] == owner, 'Foreign delivery')
    demand(all(type(v) is int for v in delivery[1:4]) and delivery[1] == tick
           and delivery[2] == source and delivery[3] == tick + 1
           and type(delivery[4]) is str and delivery[4] == 'training', 'Wrong receipt/source/delay/purpose')
    values = dict(before)
    after = dict(values)
    public_scale = 1 << (tick // 64)
    count_step = public_scale if history == 'discounted' else 1
    weight = float(public_scale) if error_rule == 'public_exponential' else 1.0
    prior = public_scale if history == 'discounted' else 1
    for route, key, side in credits:
        left, right = values[(route, 'mass_left', key)], values[(route, 'mass_right', key)]
        probability = (right + prior) / ((left + right) + (prior + prior))
        difference = probability - (0 if side == 'mass_left' else 1)
        error_step = weight * (difference ** 2)
        increments = (((route, side, key), count_step), ((route, 'observations', -1), 1),
                      ((route, 'error_sum', -1), error_step), ((route, 'weight_sum', -1), weight))
        for address, increment in increments:
            value = values[address] + increment
            demand(value > values[address], 'Absorbed, nonpositive or decreasing increment')
            stamp(address, value)
            after[address] = value
    output = tuple((address, after[address]) for address in CATALOG)
    validate_fields(output)
    return output, diagnostic, consumed | frozenset(((owner, tick),))
