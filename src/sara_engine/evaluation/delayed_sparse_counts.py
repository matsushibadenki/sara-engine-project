"""Bounded receipt-local integer updates with chronological delayed feedback."""
from dataclasses import dataclass
from datetime import datetime
from typing import Callable


@dataclass(frozen=True)
class Receipt:
    identity: int
    cutoff: datetime
    due: datetime
    addresses: tuple[str, ...]
    probability: float


class DelayedSparseCounts:
    """Three sparse count routes; no gradient, matrix operation or GPU."""

    def __init__(self, capacity=100000, pending_capacity=32):
        self.capacity = capacity
        self.pending_capacity = pending_capacity
        self.cells = {}
        self.pending = {}
        self.next_identity = 0
        self.changed_writes = 0
        self.delivered = 0
        self.peak_pending = 0
        self.time = None

    def predict(self, addresses, cutoff, due):
        if due <= cutoff or (self.time is not None and cutoff < self.time):
            raise ValueError('Noncausal prediction time')
        if len(addresses) != 3 or len(set(addresses)) != 3:
            raise ValueError('Three distinct local routes required')
        if len(self.pending) >= self.pending_capacity:
            raise ValueError('Pending receipt capacity exceeded')
        if any(r.due <= cutoff for r in self.pending.values()):
            raise ValueError('Due outcomes must be delivered before prediction')
        p = .5
        for address in addresses:
            neg, pos = self.cells.get(address, (0, 0))
            p = (pos + 8 * p) / (neg + pos + 8)
        receipt = Receipt(self.next_identity, cutoff, due, addresses, p)
        self.next_identity += 1
        self.pending[receipt.identity] = receipt
        self.peak_pending = max(self.peak_pending, len(self.pending))
        self.time = cutoff
        return receipt

    def deliver(self, receipt, at, outcome_supplier: Callable[[], int], *, training):
        if self.pending.get(receipt.identity) is not receipt or at != receipt.due:
            raise ValueError('Unknown, stale or mistimed receipt')
        if self.time is not None and at < self.time:
            raise ValueError('Outcome clock regressed')
        if any(r.due < at for r in self.pending.values()):
            raise ValueError('Earlier outcomes must be delivered first')
        missing = set(receipt.addresses) - self.cells.keys()
        if training and len(self.cells) + len(missing) > self.capacity:
            raise ValueError('Local route capacity exceeded')
        label = outcome_supplier()
        if type(label) is not int or label not in (0, 1):
            raise ValueError('Binary integer outcome required')
        if training:
            for address in receipt.addresses:
                self.cells.setdefault(address, [0, 0])[label] += 1
                self.changed_writes += 1
        del self.pending[receipt.identity]
        self.time = at
        self.delivered += 1

    def checkpoint(self):
        if self.pending:
            raise ValueError('Pending receipts cannot be saved')
        return {'schema': 1, 'capacity': self.capacity,
                'cells': {k: list(v) for k, v in self.cells.items()},
                'changed_writes': self.changed_writes}

    @classmethod
    def restore(cls, state):
        if state.get('schema') != 1:
            raise ValueError('Unknown checkpoint schema')
        model = cls(state['capacity'])
        if len(state['cells']) > model.capacity:
            raise ValueError('Checkpoint capacity exceeded')
        for key, counts in state['cells'].items():
            if not isinstance(key, str) or len(counts) != 2 or any(type(v) is not int or v < 0 for v in counts):
                raise ValueError('Invalid local count state')
            model.cells[key] = list(counts)
        writes = state['changed_writes']
        if type(writes) is not int or writes != sum(sum(v) for v in model.cells.values()):
            raise ValueError('Write accounting differs from integer changes')
        model.changed_writes = writes
        return model
