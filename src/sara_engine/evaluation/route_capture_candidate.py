"""Dormant single-owner capture candidate. Launch forbidden in this version."""
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from sara_engine.evaluation.route_owner_host import OwnerHost
from sara_engine.evaluation.route_outcome_admission import Input

LAUNCH_AUTHORIZED = False


class CaptureBlocked(ValueError):
    """No capture trial is authorized by this source candidate."""


def _primitive(value):
    """Detached typed immutable encoding, without invoking model methods/getters."""
    kind=type(value)
    if value is None:return ('none',)
    if kind is bool:return ('bool',value)
    if kind is int:return ('int',str(value))
    if kind is float:
        if not isfinite(value):raise CaptureBlocked('Nonfinite primitive state')
        return ('float',value.hex())
    if kind is str:return ('str',value)
    if kind in (tuple,list):
        return ('tuple' if kind is tuple else 'list',tuple(_primitive(item) for item in value))
    if kind is dict:
        pairs=[(_primitive(k),_primitive(v)) for k,v in value.items()]
        return ('dict',tuple(sorted(pairs,key=lambda pair:_encoded(pair[0]))))
    raise CaptureBlocked('Unsupported nonprimitive model field')


def _encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


@dataclass(frozen=True,slots=True)
class PhaseCapture:
    owner_label:str
    epoch_label:str
    revision:int
    state:tuple
    digest:str


@dataclass(frozen=True,slots=True)
class CapturePair:
    input:Input
    predecessor:PhaseCapture
    receipt:PhaseCapture


class RouteCaptureCandidate:
    """One-use candidate only; cannot instantiate while the fixed gate is false.

    This source does not establish an authenticated runner or cover external
    Python aliases. Any runtime experiment requires a separately pinned version.
    """

    def __init__(self,*,owner_label,epoch_label,seed,history,error_weight):
        if LAUNCH_AUTHORIZED is not True:
            raise CaptureBlocked('Runtime launch is not authorized in this version')
        # This import and all model operations remain unexecuted in this study.
        from sara_engine.evaluation.route_memory_factorial_v4 import FactorialMemory
        self._host,self._owner=OwnerHost.create(owner_label,epoch_label)
        self._model=FactorialMemory(seed,history,error_weight)
        self._used=False

    def _read(self,view):
        state=_primitive(vars(self._model))
        raw=_encoded(state)
        if len(raw)>262144:raise CaptureBlocked('Capture artifact limit exceeded')
        return PhaseCapture(view.owner_label,view.epoch_label,view.revision,state,sha256(raw).hexdigest())

    def _double_read(self,ticket):
        before=self._host.revalidate(self._owner,ticket)
        first=self._read(before)
        second=self._read(before)
        after=self._host.revalidate(self._owner,ticket)
        if first!=second or before!=after:raise CaptureBlocked('Independent readback differs')
        return first

    @contextmanager
    def capture(self,packet):
        if LAUNCH_AUTHORIZED is not True:
            raise CaptureBlocked('Runtime launch is not authorized in this version')
        if self._used or type(packet) is not Input:
            raise CaptureBlocked('Expected one immutable input-only envelope')
        self._used=True
        revision=self._host.inspect(self._owner).revision
        try:
            with self._host.paused(self._owner,revision) as pre_ticket:
                view=self._host.revalidate(self._owner,pre_ticket)
                if (packet.owner,packet.epoch)!=(view.owner_label,view.epoch_label):
                    raise CaptureBlocked('Input owner/epoch differs')
                if packet.receiving_id!=self._model.steps:
                    raise CaptureBlocked('Input is not the model receipt clock')
                predecessor=self._double_read(pre_ticket)
            with self._host.mutation(self._owner,revision):
                self._model.begin(packet.events,packet.receiving_id)
            with self._host.paused(self._owner,revision+2) as receipt_ticket:
                receipt=self._double_read(receipt_ticket)
                if self._model.pending is None:raise CaptureBlocked('No already-recorded receipt')
                pair=CapturePair(packet,predecessor,receipt)
                if len(_encoded((pair.predecessor.state,pair.receipt.state)))>262144:
                    raise CaptureBlocked('Paired capture artifact limit exceeded')
                yield pair
                self._host.revalidate(self._owner,receipt_ticket)
        finally:
            # Poisoned control intervals stay terminal; stable capture ends closed.
            current=self._host.inspect(self._owner)
            if current.phase=='stable':self._host.close(self._owner,current.revision)
