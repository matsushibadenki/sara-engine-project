"""Streaming, train/development-only electricity comparison."""
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
import json
from time import process_time

from sara_engine.evaluation.electricity_access_boundary import parse_support_row
from sara_engine.evaluation.electricity_learner import ARMS, ElectricityLearner, SignedElectricityLearner, encode_history, brier_score


@dataclass(frozen=True, slots=True)
class Episode:
    identity: str
    client: int
    cutoff: object
    split: str
    routes: object
    target: int
    last_direction: int


def iter_episodes(lines, protocol, *, client_count=370):
    """Use future values for eligibility/label only, never for encoding."""
    history, delay = protocol['history_observations_including_cutoff'], protocol['outcome_delay_intervals']
    if history != 96 or delay != 4:
        raise ValueError('Frozen episode geometry changed')
    window = deque(maxlen=100)
    activation = [None] * client_count
    previous = None
    excluded = set(protocol['excluded_transition_dates'])
    for line in lines:
        timestamp, values = parse_support_row(line, client_count=client_count)
        if previous is not None and timestamp - previous != timedelta(minutes=15):
            raise ValueError('Non-contiguous source timestamps')
        previous = timestamp
        if values is None:
            window.clear()
            continue
        split = 'training' if timestamp.year in protocol['training_years'] else 'development'
        for index, value in enumerate(values):
            if activation[index] is None and value is not None and value > 0:
                activation[index] = timestamp
        window.append((timestamp, values, split))
        if len(window) != 100:
            continue
        cutoff = window[95][0]
        if cutoff.strftime('%H:%M:%S') != protocol['cutoff_local_time']:
            continue
        if any(row[2] != split or row[0].date().isoformat() in excluded for row in window):
            continue
        for client in range(client_count):
            if activation[client] is None or activation[client] > window[0][0]:
                continue
            if any(row[1][client] is None for row in window):
                continue
            start, end = window[95][1][client], values[client]
            if end == start:
                continue
            # Only indices 0..95 are sent to the encoder; outcome remains separate.
            routes = encode_history((window[index][1][client] for index in range(96)), cutoff)
            yield Episode(f'{cutoff.isoformat()}:{client}', client, cutoff, split, routes,
                          int(end > start), routes.comparison_routes[-1] % 3)


class Metrics:
    """Bounded sufficient statistics, without retaining prediction rows."""
    def __init__(self):
        self.clients = defaultdict(lambda: [0.0, 0])
        self.months = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        self.classes = [[0, 0], [0, 0]]
        self.trace = sha256()

    def add(self, episode, probability):
        loss = brier_score(probability, episode.target)
        for cell in (self.clients[episode.client], self.months[episode.cutoff.month][episode.client]):
            cell[0] += loss
            cell[1] += 1
        cell = self.classes[episode.target]
        cell[0] += int(int(probability >= 0.5) == episode.target)
        cell[1] += 1
        self.trace.update(json.dumps([episode.identity, probability], separators=(',', ':')).encode() + b'\n')

    def finish(self):
        if not self.clients:
            raise ValueError('Empty development split')
        count = sum(cell[1] for cell in self.clients.values())
        return {
            'macro_client_brier': sum(total / n for total, n in self.clients.values()) / len(self.clients),
            'micro_brier': sum(cell[0] for cell in self.clients.values()) / count,
            'balanced_accuracy': (sum(correct / n for correct, n in self.classes) / 2
                                  if all(n for _, n in self.classes) else None),
            'monthly_macro_client_brier': {str(month): sum(total / n for total, n in clients.values()) / len(clients)
                                           for month, clients in sorted(self.months.items())},
            'episodes': count, 'clients': len(self.clients),
            'prediction_trace_sha256': self.trace.hexdigest(),
        }


def evaluate(episodes, *, scoring_contract='v1'):
    if scoring_contract not in ('v1', 'signed_v2'):
        raise ValueError('Unknown scoring contract')
    learner_type = SignedElectricityLearner if scoring_contract == 'signed_v2' else ElectricityLearner
    learners = {arm: learner_type(arm) for arm in ARMS}
    controls = ('training_global_prevalence', 'training_weekday_month_prevalence_with_global_fallback',
                'last_observed_direction_with_training_global_fallback_on_tie')
    metrics = {name: Metrics() for name in (*ARMS, *controls)}
    cpu = dict.fromkeys(ARMS, 0.0)
    updates = {arm: sha256() for arm in ARMS}
    peak_state = dict.fromkeys(ARMS, 0)
    global_counts = [0, 0]
    calendar_counts = defaultdict(lambda: [0, 0])
    counts = {'training': [0, 0], 'development': [0, 0]}
    development_started = False
    previous_order = None
    for number, episode in enumerate(episodes):
        order = (episode.cutoff, episode.client)
        if previous_order is not None and order <= previous_order:
            raise ValueError('Episode ordering violation')
        previous_order = order
        if episode.split not in counts:
            raise ValueError('Unknown split')
        training = episode.split == 'training'
        if training and development_started:
            raise ValueError('Training after development')
        development_started |= not training
        counts[episode.split][episode.target] += 1
        for arm, learner in learners.items():
            begin = process_time()
            receipt = learner.predict(episode.routes, time=number * 5, outcome_id=episode.identity)
            # V1 has no score-to-probability mapping. Reject signed scores
            # rather than silently changing the consumed experiment contract.
            probability = learner.probability(receipt)
            brier_score(probability, episode.target)
            update = learner.resolve(receipt, target=episode.target, time=number * 5 + 4,
                                     outcome_id=episode.identity, training=training)
            cpu[arm] += process_time() - begin
            if training:
                updates[arm].update(json.dumps([episode.identity, update.deltas], separators=(',', ':')).encode() + b'\n')
            else:
                metrics[arm].add(episode, probability)
            # State is monotone in admitted routes, but scalar sizes can vary.
            peak_state[arm] = max(peak_state[arm], learner.state_bytes())
        key = (episode.cutoff.weekday(), episode.cutoff.month)
        if training:
            global_counts[episode.target] += 1
            calendar_counts[key][episode.target] += 1
        else:
            if not sum(global_counts):
                raise ValueError('No training baseline support')
            global_probability = global_counts[1] / sum(global_counts)
            cell = calendar_counts.get(key)
            calendar_probability = cell[1] / sum(cell) if cell and sum(cell) else global_probability
            last_probability = (0.0 if episode.last_direction == 0 else
                                1.0 if episode.last_direction == 2 else global_probability)
            for name, probability in zip(controls, (global_probability, calendar_probability, last_probability)):
                metrics[name].add(episode, probability)
    results = {name: metric.finish() for name, metric in metrics.items()}
    reference = min(('scalar', *controls), key=lambda name: results[name]['macro_client_brier'])
    candidate = results['stateful_lif']
    improvement = results[reference]['macro_client_brier'] - candidate['macro_client_brier']
    months = candidate['monthly_macro_client_brier']
    ref_months = results[reference]['monthly_macro_client_brier']
    deltas = {month: ref_months[month] - score for month, score in months.items() if month in ref_months}
    parity = (results['scalar']['prediction_trace_sha256'] == results['compact_event']['prediction_trace_sha256']
              and updates['scalar'].hexdigest() == updates['compact_event'].hexdigest())
    gates = {'primary': improvement >= 0.01,
             'calendar': len(deltas) == 12 and sum(value > 0 for value in deltas.values()) >= 9
                         and min(deltas.values(), default=-1) >= -0.02,
             'causal': results['stateful_reset_each_event']['macro_client_brier'] - candidate['macro_client_brier'] >= 0.01,
             'parity': parity, 'resources': all(size <= 65536 for size in peak_state.values())}
    return {'schema': ('sara-r2-electricity-development-v2' if scoring_contract == 'signed_v2'
                       else 'sara-r2-electricity-development-v1'),
            'scoring_contract': scoring_contract,
            'adopted': all(gates.values()) if scoring_contract == 'v1' else False,
            'diagnostic_gates_passed': all(gates.values()),
            'confirmatory': scoring_contract == 'v1',
            'gates': gates, 'reference': reference, 'primary_improvement': improvement,
            'monthly_improvements': deltas, 'results': results, 'label_counts_fall_rise': counts,
            'resources': {arm: {'cpu_process_seconds': cpu[arm], 'peak_model_state_bytes': peak_state[arm],
                                'events': learner.events, 'unit_ticks': learner.ticks, 'updates': learner.updates,
                                'update_trace_sha256': updates[arm].hexdigest()}
                          for arm, learner in learners.items()},
            'held_out_numeric_cells_parsed': 0, 'energy_measured': False}
