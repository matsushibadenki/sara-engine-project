"""Preregistered synthetic shared-memory learning and backend attribution."""
from hashlib import sha256
import json
import random
from time import process_time

from sara_engine.evaluation.electricity_learner import HistoryRoutes, unit_activity
from sara_engine.evaluation.independent_temporal_design import bounded_events
from sara_engine.evaluation.temporal_information import TemporalInformationObserver
from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.learning.local_outcome import BoundedLocalOutcomeReadout, LocalOutcomeConfig


ARMS = ('observer_only', 'observer_compact', 'observer_lif', 'order_blind', 'backend_only_lif')


def episodes(seed, pairs):
    generator = random.Random(seed)
    for pair in range(pairs):
        suffix = tuple(generator.randrange(2, 24) for _ in range(91)) + (0, 1)
        modes = [0, 1]
        generator.shuffle(modes)
        for mode in modes:
            yield f'{seed}:{pair}:{mode}', ((0, 1) if mode == 0 else (1, 0)) + suffix, mode


def features(events, arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    events = bounded_events(events)
    observer = TemporalInformationObserver()
    for route in events:
        observer.observe(route)
    packet = observer.snapshot()
    active = list(packet.readout_routes()) if arm != 'backend_only_lif' else []
    if arm == 'order_blind':
        active = [(route, value) for route, value in active if route % 3 != 0]
    ticks = 0
    if arm == 'observer_compact':
        active.extend((72 + route, 1.0) for route in sorted(set(events)))
    elif arm in ('observer_lif', 'backend_only_lif'):
        admitted, ticks = unit_activity(HistoryRoutes(events, (24, 31, 43)), 'stateful_lif')
        active.extend((72 + route, value) for route, value in admitted if route < 24)
    active.append((96, 1.0))
    if len(active) > 97:
        raise ValueError('Readout capacity exceeded')
    # Include observer, packet, active channels and conservative backend scratch.
    memory = _deep_size((observer.__dict__, packet.records, active))
    if ticks:
        memory += _deep_size(([0.0] * 24, [0] * 24))
    return tuple(active), {'observer_record_visits': 95, 'observer_records': len(packet.records),
                           'observer_wire_bytes': len(packet.wire_bytes()),
                           'projection_channels': len(packet.records) * 3,
                           'backend_unit_ticks': ticks,
                           'backend_event_visits': 95 if arm == 'observer_compact' else 0,
                           'transient_bytes': memory}


def readout():
    return BoundedLocalOutcomeReadout(LocalOutcomeConfig(
        learning_rate=0.15, trace_decay=1.0, max_age=4,
        max_routes=97, max_active=97, max_delta=0.5, integral_rate=0.0))


def predict(model, active, time):
    receipt = model.predict(active, time=time)
    if not -1 <= receipt.score <= 1:
        raise ValueError('Signed score outside contract')
    return receipt, (receipt.score + 1) / 2


def evaluate_frozen(model, arm, seed, pairs, *, start_time=0):
    weights_before = model.snapshot()['weights']
    digest = sha256()
    loss = correct = count = 0
    pair_equal = True
    previous = None
    for index, (identity, events, target) in enumerate(episodes(seed, pairs)):
        active, costs = features(events, arm)
        receipt, probability = predict(model, active, start_time + index * 5)
        if _deep_size(model.__dict__) + costs['transient_bytes'] > 65536:
            raise ValueError('Evaluation state gate exceeded')
        model.discard(receipt, time=start_time + index * 5 + 4)
        loss += (probability - target) ** 2
        correct += int(int(probability >= 0.5) == target)
        count += 1
        digest.update(json.dumps([identity, probability], separators=(',', ':')).encode() + b'\n')
        if index % 2 == 0:
            previous = probability
        else:
            pair_equal &= probability == previous
    if model.snapshot()['weights'] != weights_before:
        raise ValueError('Frozen evaluation updated weights')
    return {'brier': loss / count, 'accuracy': correct / count,
            'episodes': count, 'prediction_trace_sha256': digest.hexdigest(),
            'opposite_pair_predictions_equal': pair_equal}


def run_seed(training_seed, evaluation_seed, training_pairs, evaluation_pairs):
    results, states = {}, {}
    for arm in ARMS:
        model = readout()
        cpu_start = process_time()
        totals = dict.fromkeys(('observer_record_visits', 'projection_channels', 'backend_unit_ticks', 'backend_event_visits', 'updates'), 0)
        peak = 0
        for index, (_, events, target) in enumerate(episodes(training_seed, training_pairs)):
            active, costs = features(events, arm)
            receipt, _ = predict(model, active, index * 5)
            # Receipt identity is checked by the local readout; only active routes update.
            update = model.observe(receipt, 2 * target - 1, time=index * 5 + 4)
            totals['updates'] += update.route_visits
            for name in totals:
                if name != 'updates':
                    totals[name] += costs[name]
            total_state = _deep_size(model.__dict__) + costs['transient_bytes']
            peak = max(peak, total_state)
            if total_state > 65536 or costs['backend_unit_ticks'] > 2280:
                raise ValueError('Resource gate exceeded')
        result = evaluate_frozen(model, arm, evaluation_seed, evaluation_pairs, start_time=training_pairs * 10)
        result['cpu_process_seconds_train_and_evaluation'] = process_time() - cpu_start
        # Replay from the final bounded weights, never retrain or select a candidate.
        saved = model.snapshot()['weights']
        replay = readout()
        replay._weights = dict(saved)
        replay_result = evaluate_frozen(replay, arm, evaluation_seed, evaluation_pairs)
        result['replay_equal'] = replay_result['prediction_trace_sha256'] == result['prediction_trace_sha256']
        result['peak_training_state_bytes_including_observer'] = peak
        result['training_costs'] = totals
        result['evaluation_observer_record_visits'] = 2 * evaluation_pairs * 95
        result['evaluation_backend_unit_ticks'] = 2 * evaluation_pairs * (2280 if arm in ('observer_lif', 'backend_only_lif') else 0)
        results[arm] = result
        states[arm] = {str(route): value for route, value in sorted(saved.items())}
    observer, blind = results['observer_only'], results['order_blind']
    reference = min(('observer_only', 'observer_compact'), key=lambda arm: results[arm]['brier'])
    gates = {'component_learning': observer['accuracy'] >= 0.9 and blind['brier'] - observer['brier'] >= 0.001,
             'backend_increment': results[reference]['brier'] - results['observer_lif']['brier'] >= 0.01,
             'order_blind_pair_control': blind['opposite_pair_predictions_equal'],
             'backend_only_pair_control': results['backend_only_lif']['opposite_pair_predictions_equal'],
             'deterministic_replay': all(row['replay_equal'] for row in results.values())}
    return {'training_seed': training_seed, 'evaluation_seed': evaluation_seed,
            'gates': gates, 'backend_reference': reference, 'arms': results}, states
