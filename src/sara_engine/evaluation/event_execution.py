"""Bounded research executor. Diagnostics belong to an external caller only."""
from collections import deque
import math

from sara_engine.evaluation.temporal_information import TemporalInformationObserver
from sara_engine.evaluation.temporal_capability_boundary import BoundedSequenceObserver
from sara_engine.neuro.neuron import Neuron


ARMS = ('neuron', 'scalar')


class SharedInput:
    __slots__ = ('ordinal', 'sequence', 'packet')

    def __init__(self):
        self.ordinal = TemporalInformationObserver()
        self.sequence = BoundedSequenceObserver()
        self.packet = None

    def observe(self, route, gap):
        self.ordinal.observe(route)
        self.sequence.observe(route, gap)
        info = self.ordinal.snapshot()
        history = self.sequence.snapshot()
        self.packet = (info.readout_routes(), history)
        return self.packet


class Backend:
    __slots__ = ('arm', 'cells')

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError('Unknown backend')
        self.arm, self.cells = arm, {}

    def step(self, route):
        if type(route) is not int or route not in (0, 1):
            raise ValueError('Route outside fixed contract')
        if route not in self.cells:
            self.cells[route] = Neuron(route, num_branches=1) if self.arm == 'neuron' else (0.0, 0, False)
        if self.arm == 'neuron':
            cell = self.cells[route]
            cell.add_input_to_branch(0, 0.6)
            spike = cell.step()
            return (cell.v, cell.refractory_time, spike)
        potential, remaining, _ = self.cells[route]
        if remaining:
            state = (0.0, remaining - 1, False)
        else:
            # Keep the independent scalar arithmetic and arrival-indexed clock.
            activation = 1.0 / (1.0 + math.exp(-5.0 * (abs(0.6) - 0.5)))
            output = math.copysign(activation * 2.0 * abs(0.6), 0.6)
            potential = potential * 0.95 + output
            state = (0.0, 2, True) if potential > 1.0 else (potential, 0, False)
        self.cells[route] = state
        return state


class ReceiptQueue:
    """Return functional receipts; never retain a diagnostic trace or callback."""
    __slots__ = ('pending', 'timers', 'clock')

    def __init__(self):
        self.pending, self.timers, self.clock = {}, deque(), 0

    def create(self, sequence, time, features):
        if time != self.clock or sequence in self.pending:
            raise ValueError('Noncausal or duplicate receipt')
        if len(self.pending) >= 8 or len(self.timers) >= 8:
            raise ValueError('Receipt/timer budget exceeded')
        self.pending[sequence] = (time, features)
        self.timers.append((time + 4, sequence))
        return [(time, 'created', sequence, features)]

    def notify(self, sequence, time, kind):
        if time != self.clock or kind not in ('result', 'cancel'):
            raise ValueError('Invalid external notification')
        if sequence not in self.pending:
            return [(time, 'rejected_' + kind, sequence)]
        cutoff, _ = self.pending[sequence]
        if not cutoff < time <= cutoff + 4:
            raise ValueError('Noncausal or expired receipt reached delivery')
        del self.pending[sequence]
        return [(time, 'delivered' if kind == 'result' else 'cancelled', sequence)]

    def advance(self, time, inclusive):
        if type(time) is not int or time < self.clock or type(inclusive) is not bool:
            raise ValueError('Clock must be monotone')
        rows = []
        while self.timers:
            deadline, sequence = self.timers[0]
            if deadline > time or (deadline == time and not inclusive):
                break
            if deadline < self.clock:
                raise ValueError('Overdue timer was not dispatched causally')
            self.clock = deadline
            self.timers.popleft()
            if sequence in self.pending:
                del self.pending[sequence]
                rows.append((deadline, 'expired', sequence))
            else:
                rows.append((deadline, 'stale_timer', sequence))
        self.clock = time
        return rows


class Executor:
    """Fixed two-arm experiment; callers schedule inputs and consume each batch."""
    __slots__ = ('shared', 'backends', 'queues', 'closed')

    def __init__(self):
        self.shared = SharedInput()
        self.backends = {arm: Backend(arm) for arm in ARMS}
        self.queues = {arm: ReceiptQueue() for arm in ARMS}
        self.closed = False

    def apply(self, operation):
        if self.closed:
            raise ValueError('Executor is closed')
        kind, args = operation
        if kind == 'advance':
            time, inclusive = args
            return {arm: self.queues[arm].advance(time, inclusive) for arm in ARMS}
        if kind == 'notify':
            return {arm: self.queues[arm].notify(*args) for arm in ARMS}
        if kind != 'input':
            raise ValueError('Unknown operation')
        index, time, route, gap = args
        n = self.shared.ordinal.event_count
        if (type(index) is not int or index != n or n >= 6
                or type(route) is not int or route not in (0, 1)
                or type(gap) is not int or (gap != 0 if n == 0 else not 1 <= gap <= 255)):
            raise ValueError('Input outside fixed capacity/sequence contract')
        if time != sum(item[1] for item in self.shared.sequence._events) + gap:
            raise ValueError('Input clock does not match gaps')
        for queue in self.queues.values():
            if time != queue.clock or len(queue.pending) >= 8 or len(queue.timers) >= 8:
                raise ValueError('Input clock/capacity mismatch')
        packet = self.shared.observe(route, gap)
        return {arm: self.queues[arm].create(index, time, (packet, self.backends[arm].step(route)))
                for arm in ARMS}

    def close(self):
        if any(queue.pending or queue.timers for queue in self.queues.values()):
            raise ValueError('Cannot close undrained receipts')
        for backend in self.backends.values():
            backend.cells.clear()
        self.shared.ordinal._records.clear()
        self.shared.sequence._events.clear()
        self.shared.packet = None
        self.closed = True
