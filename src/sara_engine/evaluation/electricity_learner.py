"""History-only electricity routes and bounded matched research units."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from itertools import islice
import math

from sara_engine.learning.local_outcome import BoundedLocalOutcomeReadout, LocalOutcomeConfig
from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size


ARMS = ('scalar', 'compact_event', 'stateful_lif', 'stateful_reset_each_event')


@dataclass(frozen=True, slots=True)
class HistoryRoutes:
    comparison_routes: tuple[int, ...]
    calendar_routes: tuple[int, ...]


def encode_history(history, cutoff: datetime) -> HistoryRoutes:
    """Bound input consumption before conversion; never accept an outcome."""
    if cutoff.year not in (2011, 2012, 2013):
        raise ValueError('Numeric year is not authorized')
    values = tuple(islice(iter(history), 97))
    if len(values) != 96:
        raise ValueError('Exactly 96 history observations required')
    if any(not isinstance(value, Decimal) or not value.is_finite() or value < 0 for value in values):
        raise ValueError('Invalid history observation')
    routes = tuple(3 * min(7, (index - 1) // 12)
                   + (0 if values[index] < values[index - 1] else
                      2 if values[index] > values[index - 1] else 1)
                   for index in range(1, 96))
    return HistoryRoutes(routes, (24 + cutoff.weekday(), 31 + cutoff.month - 1, 43))


def unit_activity(routes: HistoryRoutes, arm: str):
    """Return route strengths and actual tick work; cap before simulation."""
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    if len(routes.comparison_routes) != 95 or len(routes.calendar_routes) != 3:
        raise ValueError('Event budget or trace shape violated')
    if any(type(route) is not int or not 0 <= route < 24 for route in routes.comparison_routes):
        raise ValueError('Comparison route out of bounds')
    weekday, month, bias = routes.calendar_routes
    if not (type(weekday) is int and 24 <= weekday <= 30
            and type(month) is int and 31 <= month <= 42
            and type(bias) is int and bias == 43):
        raise ValueError('Calendar route out of bounds')
    active = set(routes.calendar_routes)
    if arm in ('scalar', 'compact_event'):
        active.update(routes.comparison_routes)
        ticks = 0
    else:
        potentials = [0.0] * 24
        refractory = [0] * 24
        for event in routes.comparison_routes:
            for index in range(24):
                if arm == 'stateful_reset_each_event':
                    potentials[index] = 0.0
                    refractory[index] = 0
                if refractory[index]:
                    refractory[index] -= 1
                    potentials[index] = 0.0
                    continue
                # Current goes only to the observed route; all others leak.
                potentials[index] = 0.95 * potentials[index] + (0.6 if index == event else 0.0)
                if potentials[index] >= 1.0:
                    active.add(index)
                    potentials[index] = 0.0
                    refractory[index] = 2
        ticks = 95 * 24
    return tuple((route, 1.0) for route in sorted(active)), ticks


class ElectricityLearner:
    """One pending delayed receipt; development only discards receipts."""

    def __init__(self, arm: str):
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        self.arm = arm
        self.readout = BoundedLocalOutcomeReadout(LocalOutcomeConfig(
            learning_rate=0.15, trace_decay=1.0, max_age=4, max_routes=44,
            max_active=44, max_delta=0.5, integral_rate=0.0))
        self.events = self.ticks = self.updates = 0
        self._pending = None
        self._outcome_id = None

    def predict(self, routes: HistoryRoutes, *, time: int, outcome_id: str):
        if self._pending is not None:
            raise ValueError('Resolve pending receipt first')
        if not isinstance(outcome_id, str) or not outcome_id or len(outcome_id) > 128:
            raise ValueError('Invalid bounded outcome identity')
        active, ticks = unit_activity(routes, self.arm)
        receipt = self.readout.predict(active, time=time)
        self._pending, self._outcome_id = receipt, outcome_id
        self.events += 98
        self.ticks += ticks
        return receipt

    def resolve(self, receipt, *, target: int, time: int, outcome_id: str, training: bool):
        if receipt is not self._pending or outcome_id != self._outcome_id:
            raise ValueError('Outcome receipt identity mismatch')
        if type(target) is not int or target not in (0, 1):
            raise ValueError('Binary target required')
        if time != receipt.time + 4:
            raise ValueError('Outcome must arrive four intervals later')
        if type(training) is not bool:
            raise ValueError('Training flag must be boolean')
        if training:
            update = self.readout.observe(receipt, target, time=time)
            self.updates += update.route_visits
        else:
            self.readout.discard(receipt, time=time)
            update = None
        self._pending = self._outcome_id = None
        return update

    def state_bytes(self):
        size = _deep_size(self.__dict__)
        # Transient LIF work arrays are included conservatively in the cap.
        size += _deep_size(([0.0] * 24, [0] * 24))
        if size > 65536:
            raise ValueError('Model state budget exceeded')
        return size


def brier_score(probability: float, target: int) -> float:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Probability outside unit interval')
    if type(target) is not int or target not in (0, 1):
        raise ValueError('Binary target required')
    return (probability - target) ** 2
