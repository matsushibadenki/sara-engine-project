"""Equivalent three-route capacity admission for frozen count replay."""
from sara_engine.evaluation.delayed_sparse_counts import DelayedSparseCounts


class SparseCapacityCounts(DelayedSparseCounts):
    """Preserve every update; probe only receipt-local addresses for capacity."""

    def deliver(self, receipt, at, outcome_supplier, *, training):
        if self.pending.get(receipt.identity) is not receipt or at != receipt.due:
            raise ValueError('Unknown, stale or mistimed receipt')
        if self.time is not None and at < self.time:
            raise ValueError('Outcome clock regressed')
        if any(r.due < at for r in self.pending.values()):
            raise ValueError('Earlier outcomes must be delivered first')
        # predict() guarantees three distinct addresses; no model-wide scan.
        if training:
            missing = sum(address not in self.cells for address in receipt.addresses)
            if len(self.cells) + missing > self.capacity:
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
