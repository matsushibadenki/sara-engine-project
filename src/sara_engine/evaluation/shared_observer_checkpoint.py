"""Strict research-only weight validation and frozen prediction replay."""
import math

from sara_engine.evaluation.shared_observer_learning import ARMS, readout, evaluate_frozen


def checked_weights(values):
    if type(values) is not dict or len(values) > 97:
        raise ValueError('Checkpoint route budget exceeded')
    result = {}
    for key, value in values.items():
        if type(key) is not str or not 1 <= len(key) <= 2 or not key.isascii() or not key.isdigit():
            raise ValueError('Invalid route identity')
        route = int(key)
        if str(route) != key or not 0 <= route < 97:
            raise ValueError('Invalid route range')
        if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('Invalid signed weight')
        result[route] = value
    return result


def replay_checkpoint(checkpoint, report, protocol):
    if checkpoint.get('schema') != 'sara-shared-observer-research-checkpoint-v1' or checkpoint.get('production_loading_authorized') is not False:
        raise ValueError('Unknown or production-enabled checkpoint')
    if len(checkpoint['states']) != len(protocol['training_seeds']) or len(report['runs']) != len(protocol['training_seeds']):
        raise ValueError('Checkpoint seed cardinality changed')
    checks = []
    for state, run, seed, evaluation_seed in zip(checkpoint['states'], report['runs'], protocol['training_seeds'], protocol['evaluation_seeds']):
        if state['training_seed'] != seed or run['training_seed'] != seed or set(state['weights']) != set(ARMS):
            raise ValueError('Checkpoint seed/arm identity changed')
        if run['evaluation_seed'] != evaluation_seed:
            raise ValueError('Checkpoint evaluation identity changed')
        for arm in ARMS:
            model = readout()
            model._weights = checked_weights(state['weights'][arm])
            result = evaluate_frozen(model, arm, run['evaluation_seed'], protocol['evaluation_pairs_per_seed'])
            checks.append({'training_seed': seed, 'arm': arm,
                           'matched': result['prediction_trace_sha256'] == run['arms'][arm]['prediction_trace_sha256'],
                           'episodes': result['episodes']})
    return {'schema': 'sara-shared-observer-checkpoint-verification-v1',
            'passed': all(check['matched'] for check in checks), 'checks': checks,
            'replayed_predictions': sum(check['episodes'] for check in checks),
            'retraining_executed': False, 'production_loading_authorized': False}
