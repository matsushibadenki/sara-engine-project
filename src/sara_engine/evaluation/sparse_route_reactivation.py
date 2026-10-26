"""Finite nonstationary route reactivation; no spiking or deployment claim."""
from copy import deepcopy
from hashlib import sha256
import json
import random

from sara_engine.evaluation.sparse_factor_routing import ROUTES, Sample, Task, bits, evaluate
from sara_engine.evaluation.sparse_route_budget import BudgetRouteMemory

ARMS = ('round_robin', 'permanent', 'periodic', 'error_triggered', 'shuffled_error')
PROTOCOL_SHA256 = '4270ed450df82d391cdefa2204e207f1e1a3c7d66fb8b50fe5b053db1b789f4a'


def rows(seed, task_name, changed, count=1024, frozen=False):
    """Private generator. Learners receive only Sample events/outcomes."""
    task = Task(seed, task_name)
    rng = random.Random(seed + (580000 if frozen else 570000))
    result = []
    while len(result) < count:
        raw = tuple(rng.randrange(2) for _ in range(5))
        if not frozen and raw[:3] == (1, 1, 1):
            continue
        sample = task.encode(raw[:3], raw[3:])
        if changed and (frozen or len(result) >= 512):
            a, _, _, d, e = raw
            target = (d, e, a) if task_name == 'independent' else (d ^ e, e ^ a, a)
            sample = Sample(sample.events, tuple(target[i] for i in task.outputs))
        result.append(sample)
    return result


def dependencies(seed, task_name, changed):
    task = Task(seed, task_name)
    if not changed:
        return task.provided_routes()
    keys = ((3,), (4,), (0,)) if task_name == 'independent' else ((3, 4), (0, 4), (0,))
    return tuple((tuple(sorted(task.inputs.index(i) for i in keys[target])), output)
                 for output, target in enumerate(task.outputs))


class ReactivationMemory(BudgetRouteMemory):
    """Public-time weighted local residuals; nine real route updates per receipt."""

    def __init__(self, strategy, seed):
        if strategy not in ARMS:
            raise ValueError('Unknown reactivation strategy')
        super().__init__('round_robin', seed)
        self.strategy = strategy
        self.errors = [0.0]*45
        self.weights = [0.0]*45
        self.detector = 0.25
        self.explore_until = None
        self.cooldown_until = 180
        self.starts, self.commits = [], []
        self.initial_pool = None

    def cost(self, index):
        error = self.errors[index]/self.weights[index] if self.weights[index] else 0.25
        return error + 0.005*len(self.routes[index][0])

    def shortlist(self):
        return tuple(sorted(i for group in self.groups
                            for i in sorted(group, key=lambda i: (self.cost(i), i))[:3]))

    def narrow(self):
        if self.strategy == 'round_robin':
            return
        if not self.pruned and self.steps >= 180:
            self.pool = self.shortlist()
            self.initial_pool = self.pool
            self.pruned = True
        if not self.pruned:
            return
        if self.explore_until is not None and self.steps >= self.explore_until:
            previous = self.pool
            self.pool = self.shortlist()
            self.commits.append({'tick': self.steps, 'added': sorted(set(self.pool)-set(previous)),
                                 'removed': sorted(set(previous)-set(self.pool))})
            self.explore_until = None
            self.cooldown_until = self.steps+64
            self.detector = 0.25
        if self.strategy == 'permanent' or self.explore_until is not None:
            return
        trigger = ((self.strategy == 'periodic' and self.steps in (436, 692, 948)) or
                   (self.strategy in ('error_triggered', 'shuffled_error') and
                    self.steps >= self.cooldown_until and self.detector > 0.20))
        if trigger:
            if len(self.starts) >= 64:
                raise AssertionError('Bounded lifecycle metadata exceeded')
            self.starts.append(self.steps)
            self.explore_until = self.steps+180

    def schedule(self):
        if self.pruned and self.explore_until is None:
            return self.pool
        offset = (self.steps*3) % 15
        return tuple(group[(offset+j) % 15] for group in self.groups for j in range(3))

    def begin(self, events, time):
        if type(time) is not int or time != self.steps or not 0 <= time < 1024:
            raise ValueError('Fixed finite receipt clock required')
        prediction, trace = super().begin(events, time)
        # Only already-produced prediction is retained for delayed detector credit.
        self.pending = (*self.pending, prediction)
        return prediction, trace

    def observe(self, outcome, time):
        outcome = bits(outcome, 3)
        if self.pending is None or type(time) is not int or time != self.pending[0]+1:
            raise ValueError('Outcome must follow its own receipt by one tick')
        tick, events, active, probabilities, prediction = self.pending
        weight = float(2**(tick//64))
        for i, probability in zip(active, probabilities):
            inputs, slot = self.routes[i]
            key = tuple(events[j] for j in inputs)
            count = self.tables[i].setdefault(key, [0, 0])
            p = 0.5 if probability is None else probability
            before = (count[outcome[slot]], self.observations[i], self.errors[i], self.weights[i])
            count[outcome[slot]] += 1
            self.observations[i] += 1
            self.errors[i] += weight*(p-outcome[slot])**2
            self.weights[i] += weight
            after = (count[outcome[slot]], self.observations[i], self.errors[i], self.weights[i])
            if not all(a != b for a, b in zip(before, after)):
                raise AssertionError('A registered learning-field update was a no-op')
        loss = sum(((0.5 if p is None else p)-y)**2 for p, y in zip(prediction, outcome))/3
        self.detector += (loss-self.detector)/16
        self.updates += 9
        self.steps += 1
        self.pending = None


def train(model, samples, labels):
    """Count genuine changed fields; detector/scheduler metadata are separate."""
    correct, losses, ledger = [], [], sha256()
    for tick, (row, label) in enumerate(zip(samples, labels)):
        prediction, trace = model.begin(row.events, tick)
        before = tuple(model.observations)
        selected = trace['updated_routes']
        if len(set(selected)) != 9 or any(sum(ROUTES[i][1] == slot for i in selected) != 3 for slot in range(3)):
            raise AssertionError('Nine distinct, balanced route updates required')
        correct.append(tuple(None if p is None else int(p >= 0.5) for p in prediction) == row.outcome)
        losses.append(sum(((0.5 if p is None else p)-y)**2 for p, y in zip(prediction, row.outcome))/3)
        model.observe(label, tick+1)
        deltas = tuple(a-b for a, b in zip(model.observations, before))
        if deltas != tuple(int(i in selected) for i in range(45)):
            raise AssertionError('Inactive observations changed or active update was missing')
        if any(len(table) > 2**len(ROUTES[i][0]) for i, table in enumerate(model.tables)):
            raise AssertionError('Fixed binary table capacity exceeded')
        ledger.update(json.dumps([tick, prediction, selected, deltas, model.pool, model.detector]).encode()+b'\n')
    counts = sum(sum(count) for table in model.tables for count in table.values())
    if counts != len(samples)*9 or sum(model.observations) != counts or model.updates != counts:
        raise AssertionError('Independent route-count ledger differs')
    blocks = [{'end': start+64, 'exact': sum(correct[start:start+64])/64,
               'brier': sum(losses[start:start+64])/64} for start in range(0, len(samples), 64)]
    recovery = None
    post = [b for b in blocks if b['end'] > 512]
    for previous, current in zip(post, post[1:]):
        if previous['exact'] >= 0.90 and current['exact'] >= 0.90:
            recovery = current['end']-512
            break
    return {'route_updates': counts, 'actual_learning_field_changes': counts*4,
            'count_increments': counts, 'observation_increments': counts,
            'positive_error_sum_changes': counts, 'positive_weight_sum_changes': counts,
            'tail_exact': sum(correct[-128:])/128, 'tail_brier': sum(losses[-128:])/128,
            'recovery_ticks': recovery, 'blocks': blocks,
            'update_prediction_ledger_sha256': ledger.hexdigest(),
            'exploration_starts': list(model.starts), 'commits': deepcopy(model.commits),
            'final_pool': list(model.pool), 'initial_pool': model.initial_pool,
            'unfinished_exploration': model.explore_until is not None,
            'max_possible_route_keys': 4, 'metadata_writes_matched': False}


def run_case(seed, task, condition):
    changed = condition == 'changed'
    samples = rows(seed, task, changed)
    frozen = rows(seed, task, changed, 256, frozen=True)
    arms = {}
    final_dependencies = dependencies(seed, task, changed)
    for strategy in ARMS:
        model = ReactivationMemory(strategy, seed)
        labels = [row.outcome for row in samples]
        if strategy == 'shuffled_error':
            rng = random.Random(seed+590000)
            for start in (0, 512):
                phase = labels[start:start+512]
                rng.shuffle(phase)
                labels[start:start+512] = phase
        stats = train(model, samples, labels)
        result = evaluate(model, frozen)
        stats['frozen'] = result
        stats['frozen_replay_equal'] = result == evaluate(deepcopy(model), frozen)
        stats['final_dependencies_retained'] = all(ROUTES.index(route) in model.pool for route in final_dependencies)
        stats['initial_new_dependency_absent'] = (stats['initial_pool'] is not None and
            any(ROUTES.index(route) not in stats['initial_pool'] for route in final_dependencies))
        stats['state_sha256'] = state_digest(model)
        arms[strategy] = stats
    candidate, fixed = arms['error_triggered'], arms['permanent']
    integrity = {'actual_changed_learning_fields_matched': all(a['actual_learning_field_changes'] == 36864 and a['route_updates'] == 9216 for a in arms.values()),
                 'frozen_scalar_and_packet_equivalence': all(a['frozen']['scalar_reference_equal'] and a['frozen']['packet_permutation_equal'] and a['frozen_replay_equal'] for a in arms.values())}
    if changed:
        recovery = candidate['recovery_ticks']
        gates = {'dependency_change_contrast': candidate['initial_new_dependency_absent'],
                 'tail_accuracy': candidate['tail_exact'] >= 0.90,
                 'gain_over_permanent': candidate['tail_exact']-fixed['tail_exact'] >= 0.10,
                 'recovery_within_256': recovery is not None and recovery <= 256,
                 'new_shared_dependencies_retained': candidate['final_dependencies_retained'],
                 'outcome_alignment': candidate['tail_exact']-arms['shuffled_error']['tail_exact'] >= 0.20,
                 'frozen_accuracy': candidate['frozen']['exact_accuracy'] >= 0.90}
    else:
        gates = {'false_starts': len(candidate['exploration_starts']) <= 1,
                 'stationary_noninferiority': fixed['tail_exact']-candidate['tail_exact'] <= 0.02}
    return {'seed': seed, 'task': task, 'condition': condition, 'arms': arms, 'integrity': integrity, 'component_gates': gates}


def state_digest(model):
    # Table tuple keys require a canonical explicit encoding; no repr identity.
    data = deepcopy(model.__dict__)
    data['tables'] = [[(list(k), v) for k, v in sorted(table.items())] for table in model.tables]
    return sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def summarize(cases):
    integrity = all(all(c['integrity'].values()) for c in cases)
    return {'execution_integrity_passed': integrity,
            'component_passed': integrity and all(all(c['component_gates'].values()) for c in cases),
            'failed_cases': [{'seed': c['seed'], 'task': c['task'], 'condition': c['condition'],
                              'gates': [k for k, v in c['component_gates'].items() if not v]} for c in cases if not all(c['component_gates'].values())],
            'new_real_data_reads': 0, 'resource_measurements': 0, 'test_opened': False,
            'r2_independent_usefulness_established': False, 'snn_advantage_tested': False,
            'promotion_authorized': False}
