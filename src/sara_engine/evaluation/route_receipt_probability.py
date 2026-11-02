"""Immutable nonlearning receipt snapshots and delayed-result binding."""
from dataclasses import dataclass, replace
from math import isfinite


def require(value, message):
    if not value: raise ValueError(message)


def integer(value, limit):
    require(type(value) is int and 0 <= value < limit, 'Invalid exact bounded integer')


def owner_check(owner):
    require(type(owner) is tuple and len(owner) == 6, 'Invalid immutable owner')
    seed, task, condition, history, weighting, role = owner
    require(type(seed) is int and seed == 1201, 'Unknown seed')
    for value, allowed in ((task, ('independent',)), (condition, ('changed',)),
                           (history, ('cumulative', 'discounted')),
                           (weighting, ('uniform', 'public_exponential')),
                           (role, ('candidate', 'permanent', 'shuffled_outcome'))):
        require(type(value) is str and value in allowed, 'Unknown exact owner descriptor')


def diagnostic_check(value):
    require(type(value) is tuple and len(value) == 2 and value[0] == 'diagnostic'
            and type(value[1]) is tuple and len(value[1]) == 9, 'Invalid diagnostic namespace')
    for route in value[1]: integer(route, 45)
    require(len(set(value[1])) == 9, 'Duplicate diagnostic routes')


@dataclass(frozen=True, slots=True)
class View:
    owner: tuple
    route: int
    key: int
    revision: int
    allocated: bool
    n0: int
    n1: int

    def __post_init__(self):
        owner_check(self.owner); integer(self.route, 45)
        integer(self.key, 2 if self.route < 15 else 4); integer(self.revision, 1025)
        require(type(self.allocated) is bool, 'Allocation must be explicit bool')
        integer(self.n0, 6717441); integer(self.n1, 6717441)
        require(self.allocated or (self.n0 == 0 and self.n1 == 0), 'Absent key contains positive evidence')


def probability(view, tick):
    """Read a literal view; never consume an outcome or mutate a table."""
    require(type(view) is View, 'Invalid receipt view'); integer(tick, 1024)
    if not view.allocated: return None
    prior = (1 << (tick // 64)) if view.owner[3] == 'discounted' else 1
    return (view.n1 + prior) / ((view.n0 + view.n1) + (prior + prior))


@dataclass(frozen=True, slots=True)
class Snapshot:
    owner: tuple
    receiving_id: int
    source_id: int
    view: View
    probability: object
    diagnostic: tuple

    def __post_init__(self):
        owner_check(self.owner); integer(self.receiving_id, 1024); integer(self.source_id, 1024)
        require(type(self.view) is View and self.view.owner == self.owner, 'Foreign receipt view')
        require(self.source_id // 512 == self.receiving_id // 512, 'Source crosses fixed phase')
        require(self.owner[5] == 'shuffled_outcome' or self.source_id == self.receiving_id, 'Nonshuffled source differs')
        diagnostic_check(self.diagnostic)
        expected = probability(self.view, self.receiving_id)
        if expected is None: require(self.probability is None, 'Absent probability was coerced')
        else:
            require(type(self.probability) is float and isfinite(self.probability)
                    and self.probability.hex() == expected.hex(), 'Receipt-time probability differs')


@dataclass(frozen=True, slots=True)
class State:
    owner: tuple
    tick: int
    pending: object
    consumed: frozenset
    diagnostic: tuple

    def __post_init__(self):
        owner_check(self.owner); integer(self.tick, 1025); diagnostic_check(self.diagnostic)
        require(type(self.consumed) is frozenset and len(self.consumed) <= 1024, 'Mutable/unbounded consumed ledger')
        for receiving_id in self.consumed:
            integer(receiving_id, 1024)
            require(receiving_id < self.tick, 'Consumed future/current receipt')
        require(self.pending is None or type(self.pending) is Snapshot, 'Invalid pending snapshot')
        if self.pending is not None:
            require(self.pending.owner == self.owner and self.pending.receiving_id == self.tick,
                    'Foreign pending owner or receipt clock drift')
            require(self.pending.diagnostic == self.diagnostic, 'Diagnostic changed during pending interval')


@dataclass(frozen=True, slots=True)
class Delivery:
    owner: tuple
    receiving_id: int
    source_id: int
    at: int
    side: str
    purpose: str = 'training'

    def __post_init__(self):
        owner_check(self.owner); integer(self.receiving_id, 1024); integer(self.source_id, 1024); integer(self.at, 1025)
        require(type(self.side) is str and self.side in ('mass_left', 'mass_right'), 'Unsupported outcome side')
        require(type(self.purpose) is str and self.purpose in ('training', 'evaluation'), 'Invalid delivery purpose')


def open_receipt(state, receiving_id, source_id, view):
    """Freeze allocation and probability before any result-side argument exists."""
    require(type(state) is State and state.pending is None, 'Unresolved or invalid receipt state')
    integer(receiving_id, 1024); integer(source_id, 1024)
    require(receiving_id == state.tick and receiving_id not in state.consumed, 'Wrong or duplicate receiving clock')
    require(type(view) is View and view.owner == state.owner, 'Foreign or mutable view')
    snapshot = Snapshot(state.owner, receiving_id, source_id, view,
                        probability(view, receiving_id), state.diagnostic)
    return replace(state, pending=snapshot)


def resolve(state, delivery, current_view):
    """Bind a delayed fixture result to the frozen readout, without learning."""
    require(type(state) is State and state.pending is not None, 'No own pending receipt; duplicate delivery')
    require(type(delivery) is Delivery and type(current_view) is View, 'Invalid delivery/current view')
    snapshot = state.pending
    require(delivery.owner == state.owner and current_view.owner == state.owner, 'Foreign delivery/view owner')
    require(delivery.receiving_id == snapshot.receiving_id and delivery.source_id == snapshot.source_id,
            'Receiving/source identity mismatch')
    require(delivery.purpose == 'training' and delivery.at == snapshot.receiving_id + 1, 'Wrong purpose or fixed delay')
    require((current_view.route, current_view.key) == (snapshot.view.route, snapshot.view.key), 'Foreign route/key')
    require(current_view.revision >= snapshot.view.revision, 'Backward delivery revision')
    # Delivery-time masses/allocation never select the probability or public clock.
    p = snapshot.probability
    fallback = p is None
    residual_probability = 0.5 if fallback else p
    weight = float(1 << (snapshot.receiving_id // 64)) if state.owner[4] == 'public_exponential' else 1.0
    difference = residual_probability - (0 if delivery.side == 'mass_left' else 1)
    increment = weight * (difference ** 2)
    require(isfinite(increment) and increment > 0, 'Nonpositive/nonfinite fixture residual')
    receipt = {'owner': snapshot.owner, 'route': snapshot.view.route, 'key': snapshot.view.key,
               'receiving_id': snapshot.receiving_id, 'source_id': snapshot.source_id,
               'allocated_at_receipt': snapshot.view.allocated, 'probability_at_receipt': p,
               'fallback_used': fallback, 'residual_probability': residual_probability,
               'error_weight': weight, 'error_increment': increment,
               'receipt_revision': snapshot.view.revision, 'delivery_revision': current_view.revision}
    closed = replace(state, tick=state.tick + 1, pending=None,
                     consumed=state.consumed | frozenset((snapshot.receiving_id,)))
    return closed, receipt
