"""Compose two model-free references; no numerical capture or external authority."""
from threading import Lock
from sara_engine.evaluation.route_owner_host import OwnerHost,Owner
from sara_engine.evaluation.route_outcome_admission import AdmissionGate,CaptureAck,AdmissionError


class PauseAdmission:
    """Privately retain admission capabilities and host-validated pause bindings.

    Host and owner are supplied cooperative references, not authenticated origins.
    No model, callbacks, storage or numeric snapshot traversals are accepted.
    """

    def __init__(self,host,owner,expected_ack):
        if type(host) is not OwnerHost or type(owner) is not Owner or type(expected_ack) is not CaptureAck:
            raise AdmissionError('Expected exact model-free reference types')
        view=host.inspect(owner)
        if (view.phase!='stable' or view.revision!=expected_ack.pre_revision
                or view.owner_label!=expected_ack.owner or view.epoch_label!=expected_ack.epoch
                or expected_ack.receipt_revision!=expected_ack.pre_revision+2):
            raise AdmissionError('Host origin labels or stable revision plan differ')
        self._host,self._owner,self._expected=host,owner,expected_ack
        self._gate,self._producer,self._collector=AdmissionGate.create(expected_ack)
        self._pre_view=None
        self._receipt_ticket=None
        self._stopped=False
        self._lock=Lock()

    def _request(self,operation):
        if self._stopped:
            raise AdmissionError('Composition stopped; no retry')
        try:
            return operation()
        except BaseException:
            self._stopped=True
            raise

    def _live(self,ticket,revision):
        view=self._host.revalidate(self._owner,ticket)
        if (view.revision!=revision or view.owner_label!=self._expected.owner
                or view.epoch_label!=self._expected.epoch or view.phase!='stable'):
            raise AdmissionError('Live pause does not bind the registered phase')
        return view

    def admit(self,packet):
        with self._lock:
            return self._request(lambda:self._gate.admit(self._producer,packet))

    def seal_predecessor(self,ticket):
        with self._lock:
            def operation():
                if self._pre_view is not None or self._gate.phase()!='awaiting_ack':
                    raise AdmissionError('Predecessor seal requires own admitted input exactly once')
                self._pre_view=self._live(ticket,self._expected.pre_revision)
                # Return only metadata, not a reusable authority capability.
                return self._pre_view
            return self._request(operation)

    def acknowledge(self,ticket,ack):
        with self._lock:
            def operation():
                if self._pre_view is None or self._receipt_ticket is not None:
                    raise AdmissionError('Missing predecessor seal or duplicate acknowledgement')
                self._live(ticket,self._expected.receipt_revision)
                self._gate.acknowledge(self._collector,ack)
                self._receipt_ticket=ticket
            return self._request(operation)

    def release(self,ticket,tick,outcome):
        with self._lock:
            def operation():
                if self._receipt_ticket is None or ticket is not self._receipt_ticket:
                    raise AdmissionError('Delivery requires the same acknowledged pause')
                self._live(ticket,self._expected.receipt_revision)
                return self._gate.release(self._producer,tick,outcome)
            return self._request(operation)

    def status(self):
        """Control metadata only; this observation does not validate a live pause."""
        with self._lock:
            return {'stopped':self._stopped,'admission_phase':self._gate.phase(),
                    'predecessor_sealed':self._pre_view is not None,
                    'ack_recorded':self._receipt_ticket is not None,
                    'application_authorized':False}
