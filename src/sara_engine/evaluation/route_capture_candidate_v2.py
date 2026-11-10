"""Dormant capture V2 with an internal acknowledgement broker; launch forbidden."""
from dataclasses import dataclass
from hashlib import sha256
from sara_engine.evaluation.route_capture_candidate import (
    CaptureBlocked,CapturePair,PhaseCapture,_primitive,_encoded,
)
from sara_engine.evaluation.route_capture_ack import CaptureAckBridge
from sara_engine.evaluation.route_outcome_admission import Input,CaptureAck
from sara_engine.evaluation.route_owner_host import OwnerHost

LAUNCH_AUTHORIZED = False


@dataclass(frozen=True,slots=True)
class AcknowledgedCapture:
    pair: CapturePair
    ack: CaptureAck


class RouteCaptureCandidateV2:
    """No externally supplied model/callback or exported control capability.

    This version returns finalized immutable capture/ack metadata only. It is
    not a result-delivery interface and does not authenticate a real run origin.
    """

    def __init__(self,*,owner_label,epoch_label,seed,history,error_weight):
        if LAUNCH_AUTHORIZED is not True:
            raise CaptureBlocked('Runtime launch is not authorized in candidate V2')
        # Never reached by the registered static/refusal verification.
        from sara_engine.evaluation.route_memory_factorial_v4 import FactorialMemory
        self._host,self._owner=OwnerHost.create(owner_label,epoch_label)
        self._model=FactorialMemory(seed,history,error_weight)
        self._bridge=None
        self._used=False

    def _read(self,view):
        state=_primitive(vars(self._model))
        raw=_encoded(state)
        if len(raw)>262144:raise CaptureBlocked('Phase capture limit exceeded')
        return PhaseCapture(view.owner_label,view.epoch_label,view.revision,state,sha256(raw).hexdigest())

    def _double_read(self,ticket):
        before=self._host.revalidate(self._owner,ticket)
        first=self._read(before)
        second=self._read(before)
        after=self._host.revalidate(self._owner,ticket)
        if first!=second or before!=after:raise CaptureBlocked('Independent readback differs')
        return first

    def _close_control(self):
        current=self._host.inspect(self._owner)
        if current.phase=='stable':self._host.close(self._owner,current.revision)

    def capture(self,packet):
        if LAUNCH_AUTHORIZED is not True:
            raise CaptureBlocked('Runtime launch is not authorized in candidate V2')
        if self._used or type(packet) is not Input:
            raise CaptureBlocked('Expected one immutable input-only envelope')
        self._used=True
        revision=self._host.inspect(self._owner).revision
        try:
            # The internal broker receives control references; none leave this owner.
            self._bridge=CaptureAckBridge(self._host,self._owner,packet)
            with self._host.paused(self._owner,revision) as pre_ticket:
                view=self._host.revalidate(self._owner,pre_ticket)
                if (packet.owner,packet.epoch)!=(view.owner_label,view.epoch_label):
                    raise CaptureBlocked('Input owner/epoch differs')
                if packet.receiving_id!=self._model.steps:
                    raise CaptureBlocked('Input is not the model receipt clock')
                predecessor=self._double_read(pre_ticket)
                self._bridge.seal_predecessor(pre_ticket,predecessor)
            with self._host.mutation(self._owner,revision):
                self._model.begin(packet.events,packet.receiving_id)
            with self._host.paused(self._owner,revision+2) as receipt_ticket:
                receipt=self._double_read(receipt_ticket)
                pair=CapturePair(packet,predecessor,receipt)
                if len(_encoded((pair.predecessor.state,pair.receipt.state)))>262144:
                    raise CaptureBlocked('Paired capture limit exceeded')
                ack=self._bridge.acknowledge(receipt_ticket,pair)
                self._host.revalidate(self._owner,receipt_ticket)
        finally:
            self._close_control()
        return AcknowledgedCapture(pair,ack)
