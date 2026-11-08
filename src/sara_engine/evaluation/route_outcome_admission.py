"""Model-free input/result admission reference, not a real producer firewall."""
from dataclasses import dataclass
from threading import Lock


class AdmissionError(ValueError):
    """Rejected admission, acknowledgement or delivery."""


def require(value, message):
    if not value:
        raise AdmissionError(message)


def integer(value):
    require(type(value) is int and value >= 0, 'Expected exact nonnegative integer')


def bits(value, width):
    require(type(value) is tuple and len(value) == width
            and all(type(v) is int and v in (0,1) for v in value), 'Invalid immutable binary payload')


@dataclass(frozen=True, slots=True)
class Input:
    owner: str
    epoch: str
    receiving_id: int
    source_id: int
    events: tuple

    def __post_init__(self):
        for label in (self.owner,self.epoch):
            require(type(label) is str and 0 < len(label) <= 128 and label.strip() == label, 'Invalid identity label')
        integer(self.receiving_id); integer(self.source_id); bits(self.events,5)


@dataclass(frozen=True, slots=True)
class CaptureAck:
    owner: str
    epoch: str
    receiving_id: int
    source_id: int
    pre_revision: int
    receipt_revision: int
    pre_digest: str
    receipt_digest: str
    routes: tuple

    def __post_init__(self):
        for label in (self.owner,self.epoch):
            require(type(label) is str and 0 < len(label) <= 128 and label.strip() == label,'Invalid acknowledgement label')
        for v in (self.receiving_id,self.source_id,self.pre_revision,self.receipt_revision): integer(v)
        require(self.pre_revision % 2 == 0 and self.receipt_revision > self.pre_revision
                and self.receipt_revision % 2 == 0,'Invalid stable phase revisions')
        for digest in (self.pre_digest,self.receipt_digest):
            require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),'Invalid hand digest')
        require(type(self.routes) is tuple and len(self.routes) == len(set(self.routes)) == 9
                and all(type(r) is int and 0 <= r < 45 for r in self.routes),'Incomplete ordered nine-route declaration')


@dataclass(frozen=True, slots=True)
class Delivery:
    input: Input
    capture: CaptureAck
    tick: int
    outcome: tuple


class AdmissionGate:
    """One-receipt reference gate. No model, source reader, callbacks or storage."""

    def __init__(self, expected_ack):
        require(type(expected_ack) is CaptureAck,'Expected immutable capture plan')
        self._expected = expected_ack
        self._producer, self._collector = object(),object()
        self._lock = Lock()
        self._phase = 'empty'
        self._input = None

    @classmethod
    def create(cls,expected_ack):
        gate = cls(expected_ack)
        return gate,gate._producer,gate._collector

    def _capability(self,actual,expected):
        require(actual is expected,'Foreign capability')

    def _request(self,operation):
        # Called only while holding the gate, after capability authorization.
        try:
            return operation()
        except BaseException:
            if self._phase != 'released':
                self._phase = 'stopped'
            raise

    def input_view(self):
        with self._lock:
            return self._input

    def phase(self):
        with self._lock:
            return self._phase

    def admit(self,producer,packet):
        with self._lock:
            self._capability(producer,self._producer)
            def operation():
                require(self._phase == 'empty' and type(packet) is Input,'Duplicate or invalid input')
                a = self._expected
                require((packet.owner,packet.epoch,packet.receiving_id,packet.source_id)
                        == (a.owner,a.epoch,a.receiving_id,a.source_id),'Input identity differs')
                self._input = packet
                self._phase = 'awaiting_ack'
                return packet
            return self._request(operation)

    def acknowledge(self,collector,ack):
        with self._lock:
            self._capability(collector,self._collector)
            def operation():
                require(self._phase == 'awaiting_ack','No own input or duplicate acknowledgement')
                require(type(ack) is CaptureAck and ack == self._expected,'Capture binding differs')
                self._phase = 'ready'
            return self._request(operation)

    def release(self,producer,tick,outcome):
        with self._lock:
            self._capability(producer,self._producer)
            def operation():
                # Do not inspect or retain the caller payload before acknowledgement.
                require(self._phase == 'ready','Result release requires capture acknowledgement')
                integer(tick)
                require(tick == self._expected.receiving_id + 1,'Outcome is not own one-tick delayed result')
                bits(outcome,3)
                delivery = Delivery(self._input,self._expected,tick,outcome)
                self._phase = 'released'
                return delivery
            return self._request(operation)
