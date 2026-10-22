"""Single-arm research adapter using the unchanged event execution components."""
from sara_engine.evaluation.event_execution import SharedInput, Backend, ReceiptQueue, ARMS


class SingleArmExecutor:
    """Own the full shared observer in both independently diagnosed arms."""
    __slots__ = ('shared', 'backends', 'queues', 'closed', 'arm')

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError('Unknown standalone arm')
        self.arm = arm
        self.shared = SharedInput()
        self.backends = {arm: Backend(arm)}
        self.queues = {arm: ReceiptQueue()}
        self.closed = False

    def apply(self, operation):
        if self.closed:
            raise ValueError('Executor is closed')
        kind, args = operation
        queue = self.queues[self.arm]
        if kind == 'advance':
            return {self.arm: queue.advance(*args)}
        if kind == 'notify':
            return {self.arm: queue.notify(*args)}
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
        if time != queue.clock or len(queue.pending) >= 8 or len(queue.timers) >= 8:
            raise ValueError('Input clock/capacity mismatch')
        packet = self.shared.observe(route, gap)
        state = self.backends[self.arm].step(route)
        return {self.arm: queue.create(index, time, (packet, state))}

    def close(self):
        if any(queue.pending or queue.timers for queue in self.queues.values()):
            raise ValueError('Cannot close undrained receipts')
        for backend in self.backends.values():
            backend.cells.clear()
        self.shared.ordinal._records.clear()
        self.shared.sequence._events.clear()
        self.shared.packet = None
        self.closed = True
