"""SQLite transaction boundary for sequential FactorialMemory receipts."""
from contextlib import closing
from dataclasses import asdict
from hashlib import sha256
import json
import sqlite3

from sara_engine.evaluation.route_capture_candidate import CapturePair, PhaseCapture, _primitive
from sara_engine.evaluation.route_capture_layout import canonical, decode, project, validate
from sara_engine.evaluation.route_memory_factorial_v4 import FactorialMemory
from sara_engine.evaluation.route_outcome_admission import CaptureAck, Delivery, Input, bits
from sara_engine.utils.project_paths import ensure_parent_directory


def _tuple_tree(value):
    if type(value) is list:
        return tuple(_tuple_tree(item) for item in value)
    return value


def _interrupt(stage):
    """Fault injection boundary used by process interruption tests."""


class RouteReceiptStore:
    """Persist sequential receipts and their outcomes in a private database."""

    def __init__(self, path):
        self.path = ensure_parent_directory(str(path))

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.execute('PRAGMA synchronous = FULL')
        return connection

    @staticmethod
    def _checkpoint_table(connection):
        connection.execute('CREATE TABLE IF NOT EXISTS checkpoint ('
                           'slot INTEGER PRIMARY KEY CHECK (slot = 1), revision INTEGER NOT NULL, '
                           'state BLOB NOT NULL)')

    def read_committed(self):
        """Return the latest committed phase, even while a later receipt is staged."""
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN')
            packet, ack, state = connection.execute(
                'SELECT packet, ack, state FROM receipt WHERE slot = 1').fetchone()
            identity = json.loads(packet)
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'checkpoint'").fetchone()
            checkpoint = connection.execute(
                'SELECT revision, state FROM checkpoint WHERE slot = 1').fetchone() if table else None
            if checkpoint is None:
                # Legacy databases only retain a predecessor after their first commit.
                if not connection.execute('SELECT 1 FROM consumed WHERE receiving_id = ?',
                                          (identity['receiving_id'],)).fetchone():
                    return None
                revision = json.loads(ack)['receipt_revision'] + 2
            else:
                revision, state = checkpoint
        return PhaseCapture(identity['owner'], identity['epoch'], revision,
                            _tuple_tree(json.loads(state)), sha256(state).hexdigest())

    def stage_next(self, packet):
        """Run begin on a private copy of the last committed state and stage its receipt."""
        predecessor = self.read_committed()
        if (type(packet) is not Input or predecessor is None
                or (packet.owner, packet.epoch) != (predecessor.owner_label, predecessor.epoch_label)):
            raise ValueError('Input does not bind the committed stream')
        model = FactorialMemory.__new__(FactorialMemory)
        model.__dict__.update(decode(predecessor.state))
        model.begin(packet.events, packet.receiving_id)
        state = _primitive(vars(model))
        receipt = PhaseCapture(packet.owner, packet.epoch, predecessor.revision + 2,
                               state, sha256(canonical(state)).hexdigest())
        pair = CapturePair(packet, predecessor, receipt)
        ack = CaptureAck(packet.owner, packet.epoch, packet.receiving_id, packet.source_id,
                         predecessor.revision, receipt.revision, predecessor.digest, receipt.digest,
                         model.pending[2])
        self.stage(pair, ack)
        return ack

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
            connection.execute('CREATE TABLE IF NOT EXISTS receipt (slot INTEGER PRIMARY KEY CHECK (slot = 1), '
                               'packet BLOB NOT NULL, ack BLOB NOT NULL, state BLOB NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS consumed (receiving_id INTEGER PRIMARY KEY, delivery BLOB NOT NULL)')
            self._checkpoint_table(connection)
            previous = connection.execute('SELECT packet, ack, state FROM receipt WHERE slot = 1').fetchone()
            predecessor = canonical(pair.predecessor.state)
            if previous is not None:
                previous_packet, previous_ack, previous_state = previous
                identity = json.loads(previous_packet)
                if not connection.execute('SELECT 1 FROM consumed WHERE receiving_id = ?',
                                          (identity['receiving_id'],)).fetchone():
                    raise ValueError('Previous receipt is still pending')
                if ((packet.owner, packet.epoch) != (identity['owner'], identity['epoch'])
                        or packet.receiving_id != identity['receiving_id'] + 1
                        or pair.predecessor.revision != json.loads(previous_ack)['receipt_revision'] + 2
                        or predecessor != previous_state):
                    raise ValueError('Receipt does not continue the committed state')
            connection.execute('INSERT OR IGNORE INTO checkpoint VALUES (1, ?, ?)',
                               (pair.predecessor.revision, predecessor))
            connection.execute('INSERT OR REPLACE INTO receipt VALUES (1, ?, ?, ?)',
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
            receiving_id = delivery.input.receiving_id
            identity = json.loads(packet)
            if ((delivery.input.owner, delivery.input.epoch) != (identity['owner'], identity['epoch'])
                    or delivery.tick != receiving_id + 1):
                raise ValueError('Delivery does not bind the staged receipt')
            payload = canonical(asdict(delivery))
            prior = connection.execute('SELECT delivery FROM consumed WHERE receiving_id = ?',
                                       (receiving_id,)).fetchone()
            if prior is not None:
                if prior[0] != payload:
                    raise ValueError('Conflicting duplicate delivery')
                return False
            if canonical(asdict(delivery.input)) != packet or canonical(asdict(delivery.capture)) != ack:
                raise ValueError('Delivery is out of order or does not bind the staged receipt')
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
            state = canonical(_primitive(vars(model)))
            self._checkpoint_table(connection)
            connection.execute('UPDATE receipt SET state = ? WHERE slot = 1', (state,))
            connection.execute('INSERT OR REPLACE INTO checkpoint VALUES (1, ?, ?)',
                               (delivery.capture.receipt_revision + 2, state))
            _interrupt('after_state')
            connection.execute('INSERT INTO consumed VALUES (?, ?)', (receiving_id, payload))
            _interrupt('before_commit')
        _interrupt('after_commit')
        return True

    def read(self):
        """Read state and consumption from the same committed snapshot."""
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN')
            packet, state = connection.execute(
                'SELECT packet, state FROM receipt WHERE slot = 1').fetchone()
            consumed = connection.execute(
                'SELECT 1 FROM consumed WHERE receiving_id = ?', (json.loads(packet)['receiving_id'],)).fetchone()
        return decode(_tuple_tree(json.loads(state))), bool(consumed)
