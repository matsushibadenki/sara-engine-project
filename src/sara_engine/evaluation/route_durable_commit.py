"""SQLite transaction boundary for one staged FactorialMemory receipt."""
from contextlib import closing
from dataclasses import asdict
from hashlib import sha256
import json
import sqlite3

from sara_engine.evaluation.route_capture_candidate import _primitive
from sara_engine.evaluation.route_capture_layout import canonical, decode, project, validate
from sara_engine.evaluation.route_memory_factorial_v4 import FactorialMemory
from sara_engine.evaluation.route_outcome_admission import CaptureAck, Delivery, bits
from sara_engine.utils.project_paths import ensure_parent_directory


def _tuple_tree(value):
    if type(value) is list:
        return tuple(_tuple_tree(item) for item in value)
    return value


def _interrupt(stage):
    """Fault injection boundary used by process interruption tests."""


class RouteReceiptStore:
    """Persist a staged receipt and its eventual outcome in a private database."""

    def __init__(self, path):
        self.path = ensure_parent_directory(str(path))

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.execute('PRAGMA synchronous = FULL')
        return connection

    def stage(self, pair, ack):
        projection = validate(pair)
        packet = pair.input
        expected = CaptureAck(packet.owner, packet.epoch, packet.receiving_id, packet.source_id,
                              pair.predecessor.revision, pair.receipt.revision,
                              pair.predecessor.digest, pair.receipt.digest,
                              tuple(credit[0] for credit in projection['receipt_credits']))
        if type(ack) is not CaptureAck or ack != expected:
            raise ValueError('Capture acknowledgement differs')
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('CREATE TABLE receipt (slot INTEGER PRIMARY KEY CHECK (slot = 1), '
                               'packet BLOB NOT NULL, ack BLOB NOT NULL, state BLOB NOT NULL)')
            connection.execute('CREATE TABLE consumed (receiving_id INTEGER PRIMARY KEY, delivery BLOB NOT NULL)')
            connection.execute('INSERT INTO receipt VALUES (1, ?, ?, ?)',
                               (canonical(asdict(packet)), canonical(asdict(ack)), canonical(pair.receipt.state)))

    def apply(self, delivery):
        """Return True on commit or False on an exact duplicate after commit."""
        if type(delivery) is not Delivery or type(delivery.tick) is not int:
            raise ValueError('Invalid delivery')
        bits(delivery.outcome, 3)
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            packet, ack, state = connection.execute(
                'SELECT packet, ack, state FROM receipt WHERE slot = 1').fetchone()
            if (canonical(asdict(delivery.input)) != packet
                    or canonical(asdict(delivery.capture)) != ack
                    or delivery.tick != delivery.input.receiving_id + 1):
                raise ValueError('Delivery does not bind the staged receipt')
            receiving_id = delivery.input.receiving_id
            payload = canonical(asdict(delivery))
            prior = connection.execute('SELECT delivery FROM consumed WHERE receiving_id = ?',
                                       (receiving_id,)).fetchone()
            if prior is not None:
                if prior[0] != payload:
                    raise ValueError('Conflicting duplicate delivery')
                return False
            if sha256(state).hexdigest() != delivery.capture.receipt_digest:
                raise ValueError('Stored receipt digest differs')

            before = decode(_tuple_tree(json.loads(state)))
            original = project(before)
            model = FactorialMemory.__new__(FactorialMemory)
            model.__dict__.update(before)
            updates = model.updates
            model.observe(delivery.outcome, delivery.tick)
            revised = project(vars(model))
            changed = [(address, old, new) for (address, old), (_, new)
                       in zip(original.fields, revised.fields, strict=True) if old != new]
            routes = delivery.capture.routes
            if (model.updates != updates + 9 or model.pending is not None
                    or len(changed) != 36 or {address[0] for address, _, _ in changed} != set(routes)):
                raise ValueError('Unexpected nine-route update')
            connection.execute('UPDATE receipt SET state = ? WHERE slot = 1',
                               (canonical(_primitive(vars(model))),))
            _interrupt('after_state')
            connection.execute('INSERT INTO consumed VALUES (?, ?)', (receiving_id, payload))
            _interrupt('before_commit')
        _interrupt('after_commit')
        return True

    def read(self):
        """Read state and consumption from the same committed snapshot."""
        with closing(self._connect()) as connection:
            state, consumed = connection.execute(
                'SELECT state, (SELECT COUNT(*) FROM consumed) FROM receipt WHERE slot = 1').fetchone()
        return decode(_tuple_tree(json.loads(state))), bool(consumed)
