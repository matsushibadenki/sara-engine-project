"""Fixed 2x2 local evidence/error experiment; no spike or energy claim."""
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json

from sara_engine.evaluation.sparse_route_reactivation import (
    ReactivationMemory, rows, dependencies, state_digest,
)
from sara_engine.evaluation.sparse_factor_routing import ROUTES, bits, evaluate

CELLS = tuple(product(('cumulative', 'discounted'), ('uniform', 'public_exponential')))
BOUNDARIES = (180, 512, 768, 1024)


def public_weight(tick):
    return 2 ** (tick // 64)


class FactorialMemory(ReactivationMemory):
    """Lazy local evidence discounting, independent of residual weighting."""

    def __init__(self, seed, history, error_weight):
        if (history, error_weight) not in CELLS:
            raise ValueError('Unknown fixed factor cell')
        super().__init__('error_triggered', seed)
        self.history = history
        self.error_weight = error_weight

    def probability(self, index, events):
        inputs, _ = self.routes[index]
        counts = self.tables[index].get(tuple(events[i] for i in inputs))
        if counts is None:
            return None
        # steps is the next receipt tick; frozen evaluation fixes time at 1023.
        prior = public_weight(min(self.steps, 1023)) if self.history == 'discounted' else 1
        return (counts[1] + prior) / (sum(counts) + 2 * prior)

    def observe(self, outcome, time):
        outcome = bits(outcome, 3)
        if self.pending is None or type(time) is not int or time != self.pending[0] + 1:
            raise ValueError('Outcome must follow its own receipt by one tick')
        tick, events, active, probabilities, prediction = self.pending
        count_weight = public_weight(tick) if self.history == 'discounted' else 1
        error_weight = float(public_weight(tick)) if self.error_weight == 'public_exponential' else 1.0
        for index, probability in zip(active, probabilities):
            inputs, slot = self.routes[index]
            key = tuple(events[j] for j in inputs)
            count = self.tables[index].setdefault(key, [0, 0])
            p = 0.5 if probability is None else probability
            before = (count[outcome[slot]], self.observations[index], self.errors[index], self.weights[index])
            count[outcome[slot]] += count_weight
            self.observations[index] += 1
            self.errors[index] += error_weight * (p - outcome[slot]) ** 2
            self.weights[index] += error_weight
            after = (count[outcome[slot]], self.observations[index], self.errors[index], self.weights[index])
            if not all(a != b for a, b in zip(before, after)):
                raise AssertionError('A registered learning-field update was a no-op')
        loss = sum(((0.5 if p is None else p) - y) ** 2 for p, y in zip(prediction, outcome)) / 3
        self.detector += (loss - self.detector) / 16
        self.updates += 9
        self.steps += 1
        self.pending = None


def snapshot(model, true_routes):
    before = state_digest(model)
    result = []
    scale = public_weight(min(model.steps, 1023)) if model.history == 'discounted' else 1
    for index, (inputs, output) in enumerate(ROUTES):
        rank = sorted(model.groups[output], key=lambda j: (model.cost(j), j)).index(index) + 1
        keys = []
        for key in product((0, 1), repeat=len(inputs)):
            events = [0] * 5
            for j, bit in zip(inputs, key):
                events[j] = bit
            count = model.tables[index].get(key)
            keys.append({'key': list(key), 'raw_mass': deepcopy(count),
                         'effective_mass': None if count is None else [c / scale for c in count],
                         'probability': model.probability(index, tuple(events))})
        result.append({'route': index, 'observations': model.observations[index],
                       'score': model.cost(index), 'rank': rank, 'in_pool': index in model.pool,
                       'true_dependency': index in true_routes, 'keys': keys})
    if before != state_digest(model):
        raise AssertionError('Snapshot mutated learner state')
    return {'boundary': model.steps, 'query_tick': min(model.steps, 1023), 'routes': result,
            'read_only': True}


def train(model, samples, true_routes, capture=True):
    correct, losses, snapshots = [], [], []
    ledger = sha256()
    count_mass = error_mass = 0
    for tick, row in enumerate(samples):
        prediction, trace = model.begin(row.events, tick)
        selected = trace['updated_routes']
        before_obs = tuple(model.observations)
        before_tables = deepcopy(model.tables)
        before_errors, before_weights = list(model.errors), list(model.weights)
        probabilities = model.pending[3]
        if len(set(selected)) != 9 or any(sum(ROUTES[i][1] == s for i in selected) != 3 for s in range(3)):
            raise AssertionError('Nine distinct balanced updates required')
        correct.append(tuple(None if p is None else int(p >= 0.5) for p in prediction) == row.outcome)
        losses.append(sum(((0.5 if p is None else p) - y) ** 2 for p, y in zip(prediction, row.outcome)) / 3)
        model.observe(row.outcome, tick + 1)
        cw = public_weight(tick) if model.history == 'discounted' else 1
        ew = float(public_weight(tick)) if model.error_weight == 'public_exponential' else 1.0
        expected_tables = before_tables
        for i, probability in zip(selected, probabilities):
            inputs, slot = ROUTES[i]
            key = tuple(row.events[j] for j in inputs)
            expected_tables[i].setdefault(key, [0, 0])[row.outcome[slot]] += cw
            p = 0.5 if probability is None else probability
            before_errors[i] += ew * (p - row.outcome[slot]) ** 2
            before_weights[i] += ew
        if model.tables != expected_tables or model.errors != before_errors or model.weights != before_weights:
            raise AssertionError('Actual local field deltas differ from independent receipt ledger')
        deltas = tuple(a - b for a, b in zip(model.observations, before_obs))
        if deltas != tuple(int(i in selected) for i in range(45)):
            raise AssertionError('Observation update budget differs')
        count_mass += 9 * cw
        error_mass += 9 * ew
        if sum(sum(c) for table in model.tables for c in table.values()) != count_mass:
            raise AssertionError('Evidence mass conservation failed')
        if sum(model.weights) != error_mass or sum(model.observations) != (tick + 1) * 9 or model.updates != (tick + 1) * 9:
            raise AssertionError('Independent update/weight conservation failed')
        if any(len(table) > 2 ** len(ROUTES[i][0]) for i, table in enumerate(model.tables)):
            raise AssertionError('Key capacity exceeded')
        ledger.update(json.dumps([tick, prediction, selected, deltas, model.pool, model.detector]).encode() + b'\n')
        if capture and model.steps in BOUNDARIES:
            snapshots.append(snapshot(model, true_routes))
    blocks = [{'end': start + 64, 'exact': sum(correct[start:start + 64]) / 64,
               'brier': sum(losses[start:start + 64]) / 64} for start in range(0, len(samples), 64)]
    post = [b for b in blocks if b['end'] > 512]
    recovery = next((b['end'] - 512 for a, b in zip(post, post[1:])
                     if a['exact'] >= .90 and b['exact'] >= .90), None)
    return {'route_updates': model.updates, 'actual_learning_field_changes': model.updates * 4,
            'count_increments': model.updates, 'observation_increments': model.updates,
            'positive_error_sum_changes': model.updates, 'positive_weight_sum_changes': model.updates,
            'tail_exact': sum(correct[-128:]) / 128, 'tail_brier': sum(losses[-128:]) / 128,
            'recovery_ticks': recovery, 'blocks': blocks,
            'update_prediction_ledger_sha256': ledger.hexdigest(),
            'exploration_starts': list(model.starts), 'commits': deepcopy(model.commits),
            'final_pool': list(model.pool), 'initial_pool': model.initial_pool,
            'unfinished_exploration': model.explore_until is not None,
            'max_possible_route_keys': 4, 'metadata_writes_matched': False}, snapshots


def run_case(seed, task, condition, historical):
    changed = condition == 'changed'
    samples = rows(seed, task, changed)
    frozen = rows(seed, task, changed, 256, frozen=True)
    true_routes = tuple(ROUTES.index(r) for r in dependencies(seed, task, changed))
    cells = {}
    for history, error in CELLS:
        model = FactorialMemory(seed, history, error)
        stats, snapshots = train(model, samples, true_routes)
        stats['frozen'] = evaluate_cell(model, frozen)
        stats['frozen_replay_equal'] = stats['frozen'] == evaluate_cell(deepcopy(model), frozen)
        stats['final_dependencies_retained'] = all(i in model.pool for i in true_routes)
        stats['initial_new_dependency_absent'] = any(i not in stats['initial_pool'] for i in true_routes)
        if history == 'cumulative' and error == 'public_exponential':
            # Strict JSON equality permits only the registered tuple/list layout binding.
            expected = {k: historical[k] for k in stats}
            canonical = lambda x: json.dumps(x, sort_keys=True, allow_nan=False, separators=(',', ':'))
            if canonical(stats) != canonical(expected):
                raise AssertionError('Historical baseline numerical mismatch')
        cells[history + '__' + error] = {'training': stats, 'snapshots': snapshots,
            'state_sha256': state_digest(model),
            'recovery_within_256': recovery_pass(stats['recovery_ticks'])}
    return {'seed': seed, 'task': task, 'condition': condition, 'cells': cells,
            'historical_baseline_equal': True}


def recovery_pass(value):
    return value is not None and value <= 256


def summarize(cases):
    result = {}
    for history, error in CELLS:
        name = history + '__' + error
        group = [c['cells'][name] for c in cases if c['condition'] == 'changed']
        stationary = [c['cells'][name] for c in cases if c['condition'] == 'stationary']
        result[name] = {'changed_tail_mean': sum(x['training']['tail_exact'] for x in group) / len(group),
                        'changed_frozen_mean': sum(x['training']['frozen']['exact_accuracy'] for x in group) / len(group),
                        'changed_retention_passes': sum(x['training']['final_dependencies_retained'] for x in group),
                        'changed_recovery_within_256_passes': sum(x['recovery_within_256'] for x in group),
                        'changed_recovery_ticks': [x['training']['recovery_ticks'] for x in group],
                        'stationary_tail_mean': sum(x['training']['tail_exact'] for x in stationary) / len(stationary),
                        'stationary_false_starts': sum(len(x['training']['exploration_starts']) for x in stationary)}
    contrasts = []
    for case in cases:
        cell = case['cells']
        effects = {}
        for metric in ('tail_exact', 'tail_brier', 'final_dependencies_retained'):
            val = lambda h, e: float(cell[h + '__' + e]['training'][metric])
            effects[metric] = {'count_at_uniform': val('discounted','uniform') - val('cumulative','uniform'),
                'count_at_exponential': val('discounted','public_exponential') - val('cumulative','public_exponential'),
                'error_at_cumulative': val('cumulative','public_exponential') - val('cumulative','uniform'),
                'error_at_discounted': val('discounted','public_exponential') - val('discounted','uniform')}
            effects[metric]['interaction'] = effects[metric]['count_at_exponential'] - effects[metric]['count_at_uniform']
        contrasts.append({'seed': case['seed'], 'task': case['task'], 'condition': case['condition'], 'effects': effects})
    return {'cells': result, 'paired_contrasts': contrasts, 'historical_baselines_equal': True,
            'old_failure_preserved': True, 'test_opened': False, 'promotion_authorized': False,
            'snn_advantage_tested': False, 'new_real_data_reads': 0, 'resource_measurements': 0}


def evaluate_cell(model, samples):
    """Retain the frozen evaluator; replace its cumulative-only scalar oracle."""
    result = evaluate(model, samples)
    if model.history == 'discounted':
        before = state_digest(model)
        equal = True
        prior = public_weight(1023)
        for row in samples:
            actual, trace = model.predict(row.events)
            expected = [None, None, None]
            for index in trace['selected']:
                inputs, output = ROUTES[index]
                for key, masses in model.tables[index].items():
                    if all(row.events[j] == k for j, k in zip(inputs, key)):
                        expected[output] = (masses[1] + prior) / (masses[0] + masses[1] + 2 * prior)
                        break
            equal &= actual == tuple(expected)
        if before != state_digest(model):
            raise AssertionError('Scalar oracle mutated learner')
        result['scalar_reference_equal'] = equal
    if not result['scalar_reference_equal'] or not result['packet_permutation_equal']:
        raise AssertionError('Frozen scalar/packet equality failed')
    return result
