"""Matched delayed local learning and explicitly charged research execution."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from time import process_time

from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.evaluation.temporal_boundary_learning import encode, paired_inputs
from sara_engine.learning.local_outcome import BoundedLocalOutcomeReadout, LocalOutcomeConfig
from sara_engine.utils.project_paths import ensure_parent_directory


def new_readout(protocol):
    learning = protocol['learning']
    return BoundedLocalOutcomeReadout(LocalOutcomeConfig(**{
        name: learning[name] for name in ('learning_rate', 'trace_decay', 'max_age',
                                        'max_routes', 'max_active', 'max_delta', 'integral_rate')}))


def feedback(model, receipt, target, time, protocol):
    """Require the exact registered delay in addition to receipt ownership."""
    if type(time) is not int or time != receipt.time + protocol['learning']['delay']:
        raise ValueError('Feedback must arrive at the registered delay')
    if type(target) is not int or target not in (0, 1):
        raise ValueError('Invalid binary outcome')
    return model.observe(receipt, 2 * target - 1, time=time)


def shuffled_labels(seed, family, count):
    """Freeze an independent balanced SHA-rank assignment before training."""
    if count <= 0 or count % 2:
        raise ValueError('Shuffle requires a positive even episode count')
    ranked = sorted(range(count), key=lambda i: (sha256(
        f'temporal-boundary-feedback-v1:{seed}:{family}:{i}'.encode()).digest(), i))
    labels = bytearray(count)
    for index in ranked[:count // 2]:
        labels[index] = 1
    # Include sorting keys/list and serialization overhead conservatively.
    scratch_bound = _deep_size((ranked, labels)) + count * 128
    return bytes(labels), scratch_bound


def restore_readout(protocol, weights):
    model = new_readout(protocol)
    if type(weights) is not dict or len(weights) > model.config.max_routes:
        raise ValueError('Invalid saved weight capacity')
    restored = {}
    for key, value in weights.items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError('Invalid saved route')
        route = int(key)
        if route > protocol['budgets']['max_channel_id'] or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('Invalid saved weight')
        restored[route] = float(value)
    model._weights = restored
    return model


def run_phase(protocol, model, arm, family, seed, *, training, labels=None):
    """Stream one phase; evaluator identities/outcomes never enter features."""
    start = process_time()
    split = 'training' if training else 'evaluation'
    before = dict(model._weights) if not training else None
    work = dict.fromkeys(('episodes', 'validation_event_visits', 'observation_visits',
                          'canonical_event_visits', 'projection_visits', 'wire_value_visits',
                          'feature_sort_items', 'prediction_route_visits',
                          'prediction_validation_visits', 'prediction_capacity_visits',
                          'prediction_sort_items', 'prediction_normalization_visits',
                          'update_route_visits', 'update_normalization_visits',
                          'prediction_receipts', 'discarded_receipts',
                          'instrumentation_weight_entries_scanned'), 0)
    peak = peak_wire = peak_active = 0
    loss = correct = 0
    trace = sha256()
    pair_equal = True
    budget = protocol['budgets']
    offset = model._time + protocol['learning']['episode_stride']
    for identity, pair in paired_inputs(protocol, split, seed, family):
        probabilities = []
        for mode, events in pair:
            index = work['episodes']
            active, costs = encode(events, arm)
            n = len(active)
            for key, cap in (('observation_visits', 'max_observation_visits'),
                             ('projection_visits', 'max_projection_visits'),
                             ('active_channels', 'max_active_channels'), ('wire_bytes', 'max_wire_bytes')):
                if costs[key] > budget[cap]:
                    raise ValueError('Encoding budget exceeded')
            if n > budget['max_prediction_route_visits'] or (training and n > budget['max_update_route_visits']):
                raise ValueError('Local work budget exceeded')
            # A conservative sum deliberately double-counts shared objects and
            # charges prediction/update scratch before allowing local mutation.
            state_bound = (_deep_size((model, events, active, before, labels)) + costs['transient_bytes']
                           + 2048 + n * 512)
            work['instrumentation_weight_entries_scanned'] += len(model._weights)
            if state_bound > budget['max_state_bytes_per_arm']:
                raise ValueError('Total state budget exceeded')
            now = offset + index * protocol['learning']['episode_stride']
            receipt = model.predict(active, time=now)
            probability = (receipt.score + 1) / 2
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Invalid probability')
            pending_bound = _deep_size((model, events, active, before, labels)) + costs['transient_bytes'] + 2048 + n * 512
            work['instrumentation_weight_entries_scanned'] += len(model._weights)
            peak = max(peak, state_bound, pending_bound)
            if peak > budget['max_state_bytes_per_arm'] or budget['max_pending_receipts'] < 1:
                raise ValueError('Pending state budget exceeded')
            if training:
                target = labels[index] if labels is not None else mode
                update = feedback(model, receipt, target, now + protocol['learning']['delay'], protocol)
                if update.decision != 'updated' or update.route_visits != n:
                    raise ValueError('Unexpected local update')
                work['update_route_visits'] += update.route_visits
                work['update_normalization_visits'] += n
                post_bound = _deep_size((model, events, active, update, labels)) + costs['transient_bytes'] + 2048 + n * 512
                work['instrumentation_weight_entries_scanned'] += len(model._weights)
                peak = max(peak, post_bound)
                if peak > budget['max_state_bytes_per_arm']:
                    raise ValueError('Updated state budget exceeded')
            else:
                model.discard(receipt, time=now + protocol['learning']['delay'])
                work['discarded_receipts'] += 1
                loss += (probability - mode) ** 2
                correct += int(int(probability >= 0.5) == mode)
                trace.update(json.dumps([identity, mode, probability], separators=(',', ':')).encode() + b'\n')
                probabilities.append(probability)
            work['episodes'] += 1
            work['validation_event_visits'] += len(events)
            work['observation_visits'] += costs['observation_visits']
            work['canonical_event_visits'] += len(events)
            work['projection_visits'] += costs['projection_visits']
            work['wire_value_visits'] += 2 * len(events)
            work['feature_sort_items'] += n
            for key in ('prediction_route_visits', 'prediction_validation_visits',
                        'prediction_capacity_visits', 'prediction_sort_items', 'prediction_normalization_visits'):
                work[key] += n
            work['prediction_receipts'] += 1
            peak_wire = max(peak_wire, costs['wire_bytes'])
            peak_active = max(peak_active, n)
        if not training:
            pair_equal &= probabilities[0] == probabilities[1]
    if not training and model._weights != before:
        raise ValueError('Evaluation changed frozen weights')
    work['frozen_weight_copy_and_comparison_visits'] = 0 if training else 2 * len(before)
    return {'work': work, 'peak_total_state_bound_bytes': peak,
            'peak_wire_bytes': peak_wire, 'peak_active_channels': peak_active,
            'cpu_process_seconds_including_instrumentation_and_input_generation': process_time() - start,
            'accuracy': None if training else correct / work['episodes'],
            'brier': None if training else loss / work['episodes'],
            'prediction_sha256': None if training else trace.hexdigest(),
            'opposite_pair_predictions_equal': None if training else pair_equal,
            'weights_frozen': not training, 'pending_receipts_after_phase': int(model._pending is not None)}


def run_arm(protocol, arm, family, training_seed, evaluation_seed):
    model = new_readout(protocol)
    labels = None
    setup_start = process_time()
    shuffle_bytes = 0
    if arm == 'joint_shuffled':
        labels, shuffle_bytes = shuffled_labels(training_seed, family, 2 * protocol['pairs_per_family']['training'])
        if shuffle_bytes + _deep_size(model) > protocol['budgets']['max_state_bytes_per_arm']:
            raise ValueError('Feedback setup state budget exceeded')
    setup_seconds = process_time() - setup_start
    training = run_phase(protocol, model, arm, family, training_seed, training=True, labels=labels)
    evaluation = run_phase(protocol, model, arm, family, evaluation_seed, training=False, labels=labels)
    weights = {str(route): value for route, value in sorted(model._weights.items())}
    return {'training': training, 'evaluation': evaluation,
            'feedback_schedule_sha256': sha256(labels).hexdigest() if labels is not None else None,
            'feedback_schedule_setup_cpu_seconds': setup_seconds,
            'evaluator_shuffle_scratch_bound_bytes': shuffle_bytes}, weights


def gates_for_family(arms, family):
    metrics = {name: row['evaluation'] for name, row in arms.items()}
    joint = metrics['joint']
    blind = 'gap' if family == 'middle_order' else 'ordinal'
    reference = 'ordinal' if family == 'middle_order' else 'gap'
    return {'quality': joint['accuracy'] >= 0.90 and joint['brier'] <= 0.20,
            'mechanism': all(metrics[name]['brier'] - joint['brier'] >= 0.01
                             for name in ('first_last_count', 'joint_shuffled', blind)),
            'blind_pair_equality': all(metrics[name]['opposite_pair_predictions_equal']
                                       for name in ('histogram', 'first_last_count', blind)),
            'strong_reference_superiority': metrics[reference]['brier'] - joint['brier'] >= 0.01,
            'strong_reference': reference}


def save_exclusive(path, payload):
    """Durably publish once; never reuse a partially written artifact."""
    path = Path(ensure_parent_directory(str(path)))
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256(raw).hexdigest()


def execute_once(output, prefix, checkpoint, metadata, work):
    """A consumed reservation survives exceptions, crashes and retry attempts."""
    paths = [output, checkpoint, *(str(prefix) + suffix for suffix in ('_started.json', '_failed.json', '_completed.json'))]
    for path in paths:
        ensure_parent_directory(str(path))
        if Path(path).exists():
            raise FileExistsError('Experiment identity already consumed')
    save_exclusive(str(prefix) + '_started.json', {**metadata, 'status': 'started', 'attempt_consumed': True})
    try:
        report = work()
        digest = save_exclusive(output, report)
        save_exclusive(str(prefix) + '_completed.json', {
            **metadata, 'status': 'completed', 'attempt_consumed': True,
            'report_sha256': digest, 'promotion_authorized': False})
        return report
    except BaseException as error:
        save_exclusive(str(prefix) + '_failed.json', {
            **metadata, 'status': 'failed', 'attempt_consumed': True,
            'error_type': type(error).__name__})
        raise
