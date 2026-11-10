"""Generate acknowledgement from validated records, under model-free pause control."""
from hashlib import sha256
from threading import Lock
from sara_engine.evaluation.route_capture_candidate import PhaseCapture
from sara_engine.evaluation.route_capture_layout import validate,decode,project,canonical,LayoutError
from sara_engine.evaluation.route_outcome_admission import AdmissionGate,CaptureAck,Input,AdmissionError
from sara_engine.evaluation.route_owner_host import OwnerHost,Owner


class CaptureAckBridge:
    """No expected acknowledgement argument; supplied numeric state is not authenticated.

    The private preseal and active receipt pause bind cooperative control identity.
    No model, producer, numeric updater, callbacks or persistence are accepted.
    """

    def __init__(self,host,owner,packet):
        if type(host) is not OwnerHost or type(owner) is not Owner or type(packet) is not Input:
            raise AdmissionError('Expected exact model-free control and input types')
        view=host.inspect(owner)
        if view.phase!='stable' or (view.owner_label,view.epoch_label)!=(packet.owner,packet.epoch):
            raise AdmissionError('Initial owner/epoch differs')
        self._host,self._owner,self._packet=host,owner,packet
        self._initial=view
        self._pre=None
        self._gate=None
        self._producer=None
        self._receipt_ticket=None
        self._ack=None
        self._stopped=False
        self._lock=Lock()

    def _request(self,operation):
        if self._stopped:raise AdmissionError('Bridge stopped; no retry')
        try:return operation()
        except BaseException:
            self._stopped=True
            raise

    def _live(self,ticket,revision):
        view=self._host.revalidate(self._owner,ticket)
        if (view.owner_label,view.epoch_label,view.revision,view.phase)!=(self._packet.owner,self._packet.epoch,revision,'stable'):
            raise AdmissionError('Active pause binding differs')
        return view

    def seal_predecessor(self,ticket,phase):
        with self._lock:
            def operation():
                if self._pre is not None or type(phase) is not PhaseCapture:
                    raise AdmissionError('Duplicate or invalid predecessor seal')
                view=self._live(ticket,self._initial.revision)
                if (phase.owner_label,phase.epoch_label,phase.revision)!=(view.owner_label,view.epoch_label,view.revision) or type(phase.revision) is not int:
                    raise AdmissionError('Predecessor phase identity differs')
                if len(canonical(phase.state))>262144 or sha256(canonical(phase.state)).hexdigest()!=phase.digest:
                    raise LayoutError('Predecessor digest/bound differs')
                state=decode(phase.state);project(state)
                if state['pending'] is not None or state['steps']!=self._packet.receiving_id:
                    raise LayoutError('Predecessor is not own input boundary')
                self._pre=phase
            return self._request(operation)

    def acknowledge(self,ticket,pair):
        with self._lock:
            def operation():
                if self._pre is None or self._gate is not None:
                    raise AdmissionError('Missing predecessor or duplicate acknowledgement')
                self._live(ticket,self._initial.revision+2)
                projection=validate(pair)
                if pair.input!=self._packet or canonical(pair.predecessor.state)!=canonical(self._pre.state) or pair.predecessor!=self._pre:
                    raise AdmissionError('Pair differs from admitted input/private predecessor seal')
                if pair.receipt.revision!=self._initial.revision+2:
                    raise AdmissionError('Pair receipt generation differs')
                ack=CaptureAck(pair.input.owner,pair.input.epoch,pair.input.receiving_id,pair.input.source_id,
                               pair.predecessor.revision,pair.receipt.revision,pair.predecessor.digest,pair.receipt.digest,
                               tuple(credit[0] for credit in projection['receipt_credits']))
                gate,producer,collector=AdmissionGate.create(ack)
                gate.admit(producer,self._packet)
                gate.acknowledge(collector,ack)
                self._gate,self._producer,self._ack,self._receipt_ticket=gate,producer,ack,ticket
                return ack
            return self._request(operation)

    def release(self,ticket,tick,outcome):
        with self._lock:
            def operation():
                if self._gate is None or ticket is not self._receipt_ticket:
                    raise AdmissionError('No acknowledged active receipt ticket')
                self._live(ticket,self._ack.receipt_revision)
                return self._gate.release(self._producer,tick,outcome)
            return self._request(operation)

    def status(self):
        with self._lock:
            return {'stopped':self._stopped,'predecessor_sealed':self._pre is not None,
                    'admission_phase':None if self._gate is None else self._gate.phase(),
                    'application_authorized':False,'numeric_state_authenticated':False}
