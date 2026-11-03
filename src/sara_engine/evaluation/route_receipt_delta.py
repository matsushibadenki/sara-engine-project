"""Read-only binding of frozen receipt residuals to serialized numeric deltas."""
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from sara_engine.evaluation.route_receipt_probability import View, State, Delivery, resolve, owner_check, diagnostic_check
from sara_engine.evaluation.route_numeric_delta_preflight import NumericView, NumericProposal, NumericWitness, validate_numeric
from sara_engine.evaluation.route_delta_ledger_preflight import ADDRESSES, Credit
from sara_engine.evaluation.route_interface_preflight import CaseKey, Receipt, Delivery as NumericDelivery, DiagnosticHandle

META = tuple((r, k) for r, field, k in ADDRESSES if field == 'mass_left')


def require(value, message):
    if not value: raise ValueError(message)


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def case_key(owner):
    owner_check(owner)
    return CaseKey(*owner[:5], 'independent_event_driven_scalar', owner[5])


@dataclass(frozen=True, slots=True)
class Stored:
    owner: tuple
    fields: tuple
    allocation: tuple
    diagnostic: tuple

    def __post_init__(self):
        NumericView(case_key(self.owner), self.fields); diagnostic_check(self.diagnostic)
        require(type(self.allocation) is tuple and len(self.allocation) == 150, 'Incomplete allocation catalog')
        for cell in self.allocation:
            require(type(cell) is tuple and len(cell) == 2 and type(cell[0]) is tuple
                    and len(cell[0]) == 2 and all(type(v) is int for v in cell[0])
                    and type(cell[1]) is bool, 'Invalid allocation address/flag types')
        require(tuple(a for a, _ in self.allocation) == META, 'Missing/duplicate allocation address')
        values = dict(self.fields)
        for (r, k), allocated in self.allocation:
            require(allocated or (values[(r, 'mass_left', k)] == 0 and values[(r, 'mass_right', k)] == 0),
                    'Unallocated slot contains evidence')

    def encoded(self):
        return canonical({'owner': self.owner, 'fields': self.fields, 'allocation': self.allocation, 'diagnostic': self.diagnostic})


def persisted(state):
    """Canonical JSON roundtrip in memory; does not write a file or model."""
    require(type(state) is Stored, 'Invalid stored fixture state')
    obj = json.loads(state.encoded())
    return Stored(tuple(obj['owner']), tuple((tuple(a), v) for a, v in obj['fields']),
                  tuple((tuple(a), v) for a, v in obj['allocation']),
                  (obj['diagnostic'][0], tuple(obj['diagnostic'][1])))


@dataclass(frozen=True, slots=True)
class Batch:
    trusted_before: Stored
    pending: tuple
    deliveries: tuple
    allocation_target: tuple


@dataclass(frozen=True, slots=True)
class Proposal:
    before: Stored
    after: Stored
    learning_targets: tuple
    allocation_targets: tuple
    error_claims: tuple
    changed_learning_claim: int
    changed_allocation_claim: int
    delivery_revision: int


def validate(proposal, batch):
    """Inspect declared targets and actual serialized differences without applying updates."""
    require(type(proposal) is Proposal and type(batch) is Batch and type(batch.trusted_before) is Stored, 'Invalid adapter arguments')
    before, after = persisted(proposal.before), persisted(proposal.after)
    require(before.encoded() == batch.trusted_before.encoded(), 'Stale trusted predecessor')
    require(before.owner == after.owner and before.diagnostic == after.diagnostic, 'Owner/diagnostic drift')
    require(type(proposal.delivery_revision) is int and 0 <= proposal.delivery_revision <= 1024, 'Invalid view revision')
    require(type(batch.pending) is tuple and type(batch.deliveries) is tuple
            and len(batch.pending) == len(batch.deliveries) == 9, 'Nine frozen credits required')
    require(type(proposal.error_claims) is tuple and len(proposal.error_claims) == 9, 'Wrong error claim count')
    values, flags = dict(before.fields), dict(before.allocation)
    credits = []; errors = []; identity = None
    for pending, event, claimed in zip(batch.pending, batch.deliveries, proposal.error_claims):
        require(type(pending) is State and pending.pending is not None and type(event) is Delivery, 'Missing trusted pending/delivery')
        snapshot = pending.pending
        require(pending.owner == before.owner and pending.diagnostic == before.diagnostic, 'Foreign receipt/diagnostic')
        this_id = (snapshot.receiving_id, snapshot.source_id)
        if identity is None: identity = this_id
        require(this_id == identity, 'Mixed receiving/source IDs')
        r, k = snapshot.view.route, snapshot.view.key
        live = View(before.owner, r, k, proposal.delivery_revision, flags[(r, k)],
                    values[(r, 'mass_left', k)], values[(r, 'mass_right', k)])
        closed, receipt = resolve(pending, event, live)
        error = receipt['error_increment']
        require(type(claimed) is float and isfinite(claimed) and claimed.hex() == error.hex(), 'Claim not bound to frozen receipt error')
        require(closed.diagnostic == before.diagnostic, 'Resolution changed diagnostic')
        credits.append(Credit(r, k, event.side)); errors.append(error)
    require(len({c.route for c in credits}) == 9, 'Repeated route credit')
    targets = tuple(sorted([(c.route, c.mass_side, c.key_slot) for c in credits]
                           + [(c.route, field, -1) for c in credits for field in ('observations', 'error_sum', 'weight_sum')]))
    require(type(proposal.learning_targets) is tuple and all(type(a) is tuple and len(a) == 3
            and type(a[0]) is int and type(a[1]) is str and type(a[2]) is int for a in proposal.learning_targets)
            and proposal.learning_targets == targets,
            'Learning request roster differs or includes metadata/no-op targets')
    require(type(batch.allocation_target) is tuple and len(batch.allocation_target) == 2
            and all(type(v) is int for v in batch.allocation_target) and batch.allocation_target in [(c.route, c.key_slot) for c in credits], 'Foreign allocation target')
    require(type(proposal.allocation_targets) is tuple and all(type(a) is tuple and len(a) == 2
            and all(type(v) is int for v in a) for a in proposal.allocation_targets)
            and proposal.allocation_targets == (batch.allocation_target,), 'Allocation request roster differs')
    old_flags, new_flags = dict(before.allocation), dict(after.allocation)
    for address in META:
        require(new_flags[address] is (True if address == batch.allocation_target else old_flags[address]), 'Allocation admission/unchanged metadata differs')
    key = case_key(before.owner); tick, source = identity
    active = tuple(c.route for c in credits)
    receipt = Receipt(key, tick, source, active, tuple(active[j] for j in (0, 3, 6)))
    diagnostic = DiagnosticHandle(sha256(canonical(before.diagnostic)).hexdigest(), before.diagnostic[1])
    old_numeric, new_numeric = NumericView(key, before.fields), NumericView(key, after.fields)
    numeric = NumericProposal(receipt, NumericDelivery(key, tick, source, tick + 1), tuple(credits),
                              old_numeric, new_numeric, diagnostic, diagnostic)
    witness = NumericWitness(receipt, tuple(credits), tuple(errors))
    checked = validate_numeric(numeric, old_numeric, witness)
    # Count real typed differences after serialization, independent of the request list.
    changed = sum(type(a) is not type(b) or (a.hex() != b.hex() if type(a) is float else a != b)
                  for (_, a), (_, b) in zip(before.fields, after.fields))
    changed_meta = sum(a is not b for (_, a), (_, b) in zip(before.allocation, after.allocation))
    require(type(proposal.changed_learning_claim) is int and proposal.changed_learning_claim == changed == 36, 'Wrong actual learning-field claim')
    require(type(proposal.changed_allocation_claim) is int and proposal.changed_allocation_claim == changed_meta, 'Wrong metadata change claim')
    return {'requested_learning': len(targets), 'serialized_changed_learning': changed,
            'unchanged_learning': 435 - changed, 'requested_allocation': 1,
            'serialized_changed_allocation': changed_meta, 'unchanged_allocation': 150 - changed_meta,
            'frozen_error_witnesses': errors, 'numeric_validator_checks': checked,
            'before_sha256': sha256(before.encoded()).hexdigest(), 'after_sha256': sha256(after.encoded()).hexdigest(),
            'actual_runtime_learning_writes_observed': False, 'model_updates_applied': 0}
