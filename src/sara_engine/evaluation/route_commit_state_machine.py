"""Pure hand-metadata commit/consume model; never touches model or storage."""
from dataclasses import dataclass
from hashlib import sha256
import json


def require(value, message):
    if not value: raise ValueError(message)


def integer(value, limit):
    require(type(value) is int and 0 <= value < limit, 'Invalid exact bounded integer')


def digest(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Invalid opaque digest')


def owner_check(owner):
    require(type(owner) is tuple and len(owner) == 7 and type(owner[0]) is int and owner[0] == 1201, 'Invalid owner')
    for value, allowed in zip(owner[1:], (('independent',), ('changed',), ('cumulative',), ('uniform',),
                             ('local_sparse_reference', 'independent_event_driven_scalar'),
                             ('candidate', 'permanent', 'shuffled_outcome'))):
        require(type(value) is str and value in allowed, 'Foreign hand-fixture owner')


def route_check(routes):
    require(type(routes) is tuple and len(routes) == 9, 'Complete immutable nine-route batch required')
    for route in routes: integer(route, 45)
    require(len(set(routes)) == 9, 'Duplicate route credit')


@dataclass(frozen=True, slots=True)
class MetadataState:
    owner: tuple
    revision: int
    core: str
    allocation: str
    diagnostic: str
    consumed: frozenset
    last_receipt: object
    route_mask: int

    def __post_init__(self):
        owner_check(self.owner); integer(self.revision, 1025)
        digest(self.core); digest(self.allocation); digest(self.diagnostic)
        require(type(self.consumed) is frozenset and len(self.consumed) <= 1024, 'Invalid immutable consumed ledger')
        for receiving_id in self.consumed: integer(receiving_id, 1024)
        if self.last_receipt is not None: integer(self.last_receipt, 1024)
        integer(self.route_mask, 512)
        # Typed inconsistent observations remain representable for recovery classification.


def state_record(state):
    require(type(state) is MetadataState, 'Invalid observed metadata state')
    return {'owner': state.owner, 'revision': state.revision, 'core': state.core, 'allocation': state.allocation,
            'diagnostic': state.diagnostic, 'consumed': sorted(state.consumed),
            'last_receipt': state.last_receipt, 'route_mask': state.route_mask}


def fingerprint(state):
    return sha256(json.dumps(state_record(state), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Pending:
    predecessor: MetadataState
    receiving_id: int
    source_id: int
    routes: tuple
    successor_core: str
    successor_allocation: str

    def __post_init__(self):
        require(type(self.predecessor) is MetadataState, 'Invalid predecessor authority')
        integer(self.receiving_id, 1024); integer(self.source_id, 1024); route_check(self.routes)
        digest(self.successor_core); digest(self.successor_allocation)
        pre = self.predecessor
        require(self.receiving_id not in pre.consumed, 'Receipt already consumed in predecessor')
        require(self.source_id // 512 == self.receiving_id // 512, 'Source crosses original phase')
        require(pre.owner[-1] == 'shuffled_outcome' or self.source_id == self.receiving_id, 'Nonshuffled source mismatch')
        if pre.last_receipt is None:
            require(not pre.consumed and pre.route_mask == 0, 'Inconsistent trusted predecessor')
        else:
            require(pre.last_receipt in pre.consumed and pre.route_mask == 511, 'Partial trusted predecessor')


@dataclass(frozen=True, slots=True)
class Delivery:
    owner: tuple
    receiving_id: int
    source_id: int
    at: int

    def __post_init__(self):
        owner_check(self.owner); integer(self.receiving_id, 1024); integer(self.source_id, 1024); integer(self.at, 1025)


@dataclass(frozen=True, slots=True)
class Ticket:
    pending: Pending
    delivery: Delivery
    proposed_routes: tuple
    predecessor_digest: str

    def __post_init__(self):
        require(type(self.pending) is Pending, 'Invalid pending ticket')
        require(type(self.delivery) is Delivery and self.delivery.owner == self.pending.predecessor.owner
                and self.delivery.receiving_id == self.pending.receiving_id
                and self.delivery.source_id == self.pending.source_id
                and self.delivery.at == self.pending.receiving_id + 1, 'Ticket lacks bound delayed delivery')
        route_check(self.proposed_routes)
        require(self.proposed_routes == self.pending.routes, 'Partial/reordered/foreign route batch')
        digest(self.predecessor_digest)
        require(self.predecessor_digest == fingerprint(self.pending.predecessor), 'Forged predecessor digest')


def minimum_check(pending, minimum_revision):
    require(type(pending) is Pending, 'Invalid pending authority')
    integer(minimum_revision, 1025)
    require(minimum_revision >= pending.predecessor.revision, 'Minimum revision below trusted predecessor')


def ready(current, pending, minimum_revision):
    require(type(current) is MetadataState, 'Invalid current metadata')
    minimum_check(pending, minimum_revision)
    require(current.owner == pending.predecessor.owner, 'Foreign owner')
    require(pending.receiving_id not in current.consumed, 'Duplicate consumed receipt')
    require(current.revision >= minimum_revision and current == pending.predecessor, 'Stale or changed predecessor')


def prepare(current, pending, delivery, proposed_routes, minimum_revision):
    """Repeatable read-only validation; preparation does not consume a receipt."""
    ready(current, pending, minimum_revision)
    require(type(delivery) is Delivery and delivery.owner == current.owner, 'Foreign delivery')
    require(delivery.receiving_id == pending.receiving_id and delivery.source_id == pending.source_id,
            'Receiving/source identity mismatch')
    require(delivery.at == pending.receiving_id + 1, 'Wrong fixed receiving-clock delay')
    return Ticket(pending, delivery, proposed_routes, fingerprint(current))


def successor(pending):
    pre = pending.predecessor
    return MetadataState(pre.owner, pre.revision + 1, pending.successor_core, pending.successor_allocation,
                         pre.diagnostic, pre.consumed | frozenset((pending.receiving_id,)), pending.receiving_id, 511)


def commit(current, pending, ticket, minimum_revision):
    """Return one full metadata successor; this is not a storage transaction."""
    ready(current, pending, minimum_revision)
    require(type(ticket) is Ticket and ticket.pending == pending, 'Ticket differs from trusted pending authority')
    require(ticket.predecessor_digest == fingerprint(current) and ticket.proposed_routes == pending.routes,
            'Ticket predecessor/route mismatch')
    return successor(pending)


def classify(observed, pending, minimum_revision):
    """Classify hand observations; never recover, commit or authorize a retry."""
    require(type(observed) is MetadataState, 'Invalid recovery observation')
    minimum_check(pending, minimum_revision)
    if observed.owner != pending.predecessor.owner: return 'reject_inconsistent'
    if observed.revision < minimum_revision: return 'reject_stale'
    if observed == pending.predecessor: return 'predecessor_unconsumed'
    if observed == successor(pending): return 'successor_consumed'
    return 'reject_inconsistent'


def inspect_readonly(observed, pending, minimum_revision):
    return {'classification': classify(observed, pending, minimum_revision),
            'observed_digest': fingerprint(observed), 'application_authorized': False,
            'model_updates_applied': 0}
