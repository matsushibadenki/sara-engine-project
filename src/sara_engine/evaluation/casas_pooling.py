"""Fixed four-branch temporal pooling within the exact current context."""
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from heapq import heappop, heappush
import json
from sara_engine.evaluation.casas_longitudinal import addresses, metrics
from sara_engine.evaluation.casas_evidence import evidence, REFERENCES
from sara_engine.evaluation.sparse_count_replay import SparseCapacityCounts

ARMS = REFERENCES + ('pool', 'pool_evidence', 'joint_evidence', 'pool_order',
                     'pool_timing', 'pool_constant', 'shuffled_pool', 'scalar_pool')
POLICY = 'same_sensor_fast10_four_branches_v1'
GATED = ('pool_evidence', 'joint_evidence', 'pool_constant', 'shuffled_pool', 'scalar_pool')


def routes(row, arm):
    if arm in REFERENCES or arm == 'joint_evidence':
        return addresses(row, arm)
    base = addresses(row, 'current')
    same = row.history[-2][0] == row.history[-1][0]
    fast = row.gaps[-1] < 2  # Existing bins 0/1 mean <10 seconds.
    if arm in ('pool_order', 'pool_constant'):
        fast = False
    if arm == 'pool_timing':
        same = False
    return base[:2] + (repr(('pool', base[1], same, fast)),)


class PoolCounts(SparseCapacityCounts):
    """Pool by causal features, with no learned routing or extra update writes."""
    def __init__(self, arm, capacity=100000):
        if arm not in ARMS:
            raise ValueError('Unknown pooling arm')
        super().__init__(capacity)
        self.arm = arm
        self.overrides = self.fallbacks = 0
        self.active_days = set()

    def predict_example(self, row):
        keys = routes(row, self.arm)
        active = evidence(self.cells.get(keys[-1], (0,0)), self.cells.get(keys[1], (0,0))) if self.arm in GATED else False
        receipt = super().predict(keys, row.cutoff, row.due)
        if self.arm in GATED:
            if active:
                self.overrides += 1
                self.active_days.add(row.cutoff.date().isoformat())
            else:
                probability = .5
                for key in (keys[0], keys[1], keys[1]):
                    neg, pos = self.cells.get(key, (0,0))
                    probability = (pos+8*probability)/(neg+pos+8)
                receipt = replace(receipt, probability=probability)
                self.pending[receipt.identity] = receipt
                self.fallbacks += 1
        return receipt

    def snapshot(self):
        return {'arm': self.arm, 'policy': POLICY, 'counts': self.checkpoint()}

    @classmethod
    def from_snapshot(cls, state):
        if state['policy'] != POLICY:
            raise ValueError('Unknown pooling policy')
        validated = SparseCapacityCounts.restore(state['counts'])
        model = cls(state['arm'], validated.capacity)
        model.cells, model.changed_writes = validated.cells, validated.changed_writes
        return model


def phase(model, rows, training):
    queue, predictions = [], []
    labels = [r.label for r in rows]
    if training and model.arm == 'shuffled_pool':
        k = len(labels)//2
        labels = labels[k:] + labels[:k]
    def flush(at):
        while queue and queue[0][0] <= at:
            due, _, receipt, label = heappop(queue)
            model.deliver(receipt, due, lambda:label, training=training)
    for i, row in enumerate(rows):
        flush(row.cutoff)
        receipt = model.predict_example(row)
        predictions.append(receipt.probability)
        heappush(queue, (row.due, i, receipt, labels[i]))
    if queue:
        flush(max(item[0] for item in queue))
    return predictions


def identification(model, rows):
    """Post-fit descriptive evidence; never modify routes or select a threshold."""
    groups = defaultdict(dict)
    for row in rows:
        keys = routes(row, 'pool_evidence')
        groups[keys[1]][keys[-1]] = None
    opposing, maximum, branch_count = 0, 0, 0
    for current_key, branches in groups.items():
        current = model.cells[current_key]
        signs = set()
        maximum = max(maximum, len(branches))
        branch_count += len(branches)
        for key in branches:
            detail = model.cells[key]
            if evidence(detail, current):
                sign = 1 if detail[1]/sum(detail) > current[1]/sum(current) else -1
                if detail[1 if sign > 0 else 0] >= 4:
                    signs.add(sign)
        opposing += signs == {-1, 1}
    return {'opposing_evidence_contexts': opposing, 'current_contexts': len(groups),
            'pooled_branches': branch_count, 'maximum_branches_per_current': maximum,
            'route_ceiling': 1+5*len(groups), 'actual_routes': len(model.cells)}


def compare(examples):
    reports, states, traces = {}, {}, {}
    identity = None
    for arm in ARMS:
        model = PoolCounts(arm)
        phase(model, examples['train'], True)
        state = model.snapshot()
        if arm == 'pool_evidence':
            identity = identification(model, examples['train'])
        restored = PoolCounts.from_snapshot(state)
        model.overrides = model.fallbacks = 0
        model.active_days.clear()
        predictions = phase(model, examples['development'], False)
        if predictions != phase(restored, examples['development'], False) or model.snapshot() != state:
            raise ValueError('Checkpoint or frozen replay changed')
        if model.changed_writes != 3*len(examples['train']):
            raise ValueError('Actual write budget differs')
        reports[arm] = {**metrics(examples['development'], predictions),
                       'changed_writes': model.changed_writes, 'confirmation_writes': 0,
                       'routes': len(model.cells), 'peak_pending': model.peak_pending,
                       'overrides': model.overrides, 'fallbacks': model.fallbacks,
                       'active_days': len(model.active_days)}
        states[arm] = state
        traces[arm] = sha256(json.dumps(predictions).encode()).hexdigest()
    candidate = reports['pool_evidence']
    feasibility = {
        'opposing_contexts': identity['opposing_evidence_contexts'] >= 10,
        'coverage': candidate['overrides'] >= .01*len(examples['development']),
        'active_days': candidate['active_days'] >= 10,
        'bounded_pooling': identity['maximum_branches_per_current'] <= 4 and identity['actual_routes'] <= identity['route_ceiling'],
    }
    quality = {
        'brier_gain': min(reports[a]['daily_brier'] for a in REFERENCES)-candidate['daily_brier'] >= .01,
        'balanced_gain': candidate['balanced_accuracy']-max(reports[a]['balanced_accuracy'] for a in REFERENCES) >= .02,
        'gap_necessity': reports['pool_constant']['daily_brier']-candidate['daily_brier'] >= .005,
        'paired_outcome': reports['shuffled_pool']['daily_brier']-candidate['daily_brier'] >= .01,
        'scalar_equivalence': traces['pool_evidence'] == traces['scalar_pool'],
    }
    return {'arms': reports, 'traces': traces, 'identification': identity,
            'feasibility_gates': feasibility, 'quality_gates': quality,
            'passed': all(feasibility.values()) and all(quality.values()),
            'scope': 'Reused HH104 fit period only; no fresh generalization evidence',
            'previous_confirmation_opened': False, 'original_development_opened': False,
            'test_opened': False}, states
