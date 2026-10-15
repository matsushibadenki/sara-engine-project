"""Current-relative evidence from disjoint local Bernoulli count cells."""
from dataclasses import replace
from hashlib import sha256
from heapq import heappush, heappop
import json
import math
from sara_engine.evaluation.casas_longitudinal import addresses, metrics
from sara_engine.evaluation.sparse_count_replay import SparseCapacityCounts

REFERENCES = ('current', 'current_fine', 'histogram', 'transition', 'joint')
ARMS = REFERENCES + ('confidence', 'evidence', 'constant_gap', 'evidence_constant',
                     'shuffled_evidence', 'scalar_evidence')
POLICY = 'disjoint_hoeffding95_v1'


def evidence(detail, current):
    """Fixed two-sided separation proxy; serial dependence precludes a p-value claim."""
    complement = tuple(c-d for c, d in zip(current, detail))
    if len(detail) != 2 or len(current) != 2 or min(*detail, *complement) < 0:
        raise ValueError('Detail must be a subset of current counts')
    n, m = sum(detail), sum(complement)
    if not n or not m:
        return False
    # Union bound for four one-sided tails under independent Bernoulli sampling.
    radius = math.sqrt(math.log(80)/(2*n)) + math.sqrt(math.log(80)/(2*m))
    return abs(detail[1]/n - complement[1]/m) > radius


class EvidenceCounts(SparseCapacityCounts):
    """Predict with joint detail only when its local evidence rule passes."""
    def __init__(self, arm, capacity=100000):
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        super().__init__(capacity)
        self.arm = arm
        self.overrides = self.fallbacks = 0

    def predict_example(self, example):
        kind = 'constant_gap' if self.arm in ('constant_gap', 'evidence_constant') else self.arm
        routes = addresses(example, kind)
        detail = self.cells.get(routes[-1], (0, 0))
        current = self.cells.get(routes[1], (0, 0))
        gated = self.arm in ('confidence', 'evidence', 'evidence_constant',
                             'shuffled_evidence', 'scalar_evidence')
        active = sum(detail) >= 8 if self.arm == 'confidence' else evidence(detail, current)
        receipt = super().predict(routes, example.cutoff, example.due)
        if gated:
            if active:
                self.overrides += 1
            else:
                p = .5
                for key in (routes[0], routes[1], routes[1]):
                    neg, pos = self.cells.get(key, (0, 0))
                    p = (pos + 8*p)/(neg + pos + 8)
                receipt = replace(receipt, probability=p)
                self.pending[receipt.identity] = receipt
                self.fallbacks += 1
        return receipt

    def snapshot(self):
        return {'arm': self.arm, 'policy': POLICY, 'counts': self.checkpoint()}

    @classmethod
    def from_snapshot(cls, state):
        if state['policy'] != POLICY:
            raise ValueError('Unsupported evidence policy')
        validated = SparseCapacityCounts.restore(state['counts'])
        result = cls(state['arm'], validated.capacity)
        result.cells = validated.cells
        result.changed_writes = validated.changed_writes
        return result


def phase(model, rows, training):
    queue, predictions = [], []
    labels = [e.label for e in rows]
    if training and model.arm == 'shuffled_evidence':
        offset = len(labels)//2
        labels = labels[offset:] + labels[:offset]
    def flush(at):
        while queue and queue[0][0] <= at:
            due, _, receipt, label = heappop(queue)
            model.deliver(receipt, due, lambda: label, training=training)
    for i, row in enumerate(rows):
        flush(row.cutoff)
        receipt = model.predict_example(row)
        predictions.append(receipt.probability)
        heappush(queue, (row.due, i, receipt, labels[i]))
    if queue:
        flush(max(item[0] for item in queue))
    return predictions


def compare(examples):
    reports, states, traces = {}, {}, {}
    for arm in ARMS:
        model = EvidenceCounts(arm)
        phase(model, examples['train'], True)
        state = model.snapshot()
        restored = EvidenceCounts.from_snapshot(state)
        model.overrides = model.fallbacks = 0
        predictions = phase(model, examples['development'], False)
        replay = phase(restored, examples['development'], False)
        if predictions != replay or model.snapshot() != state:
            raise ValueError('Frozen checkpoint or replay changed')
        if model.changed_writes != 3*len(examples['train']):
            raise ValueError('Actual local writes differ')
        reports[arm] = {**metrics(examples['development'], predictions),
                        'changed_writes': model.changed_writes, 'confirmation_writes': 0,
                        'routes': len(model.cells), 'peak_pending': model.peak_pending,
                        'overrides': model.overrides, 'fallbacks': model.fallbacks}
        states[arm] = state
        traces[arm] = sha256(json.dumps(predictions).encode()).hexdigest()
    candidate = reports['evidence']
    gates = {
        'brier_gain': min(reports[a]['daily_brier'] for a in REFERENCES)-candidate['daily_brier'] >= .01,
        'balanced_gain': candidate['balanced_accuracy']-max(reports[a]['balanced_accuracy'] for a in REFERENCES) >= .02,
        'gap_necessity': reports['evidence_constant']['daily_brier']-candidate['daily_brier'] >= .005,
        'paired_outcome': reports['shuffled_evidence']['daily_brier']-candidate['daily_brier'] >= .01,
        'scalar_equivalence': traces['evidence'] == traces['scalar_evidence'],
    }
    return {'arms': reports, 'traces': traces, 'gates': gates, 'passed': all(gates.values()),
            'original_development_opened': False, 'test_opened': False}, states
