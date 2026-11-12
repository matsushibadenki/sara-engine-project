"""Accept one authenticated route input or delivery over standard input."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import sys

from sara_engine.evaluation.route_delivery_origin import RouteOriginKeys
from sara_engine.evaluation.route_durable_commit import RouteReceiptStore
from sara_engine.evaluation.route_outcome_admission import CaptureAck, Delivery, Input

MAX_MESSAGE_BYTES = 16384


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result


def _packet(value):
    if type(value) is not dict or type(value.get('events')) is not list:
        raise ValueError('Invalid input packet')
    return Input(**dict(value, events=tuple(value['events'])))


def dispatch(store, message):
    if type(message) is not dict or set(message) != {'operation', 'payload', 'tag'}:
        raise ValueError('Invalid message envelope')
    if type(message['tag']) is not str or len(message['tag']) != 64:
        raise ValueError('Invalid tag encoding')
    tag = bytes.fromhex(message['tag'])
    payload = message['payload']
    if message['operation'] == 'stage':
        ack = store.stage_next(_packet(payload), input_tag=tag)
        return {'ack': asdict(ack)}
    if message['operation'] == 'apply':
        if type(payload) is not dict or set(payload) != {'input', 'capture', 'tick', 'outcome'}:
            raise ValueError('Invalid delivery payload')
        capture = payload['capture']
        if (type(capture) is not dict or type(capture.get('routes')) is not list
                or type(payload['outcome']) is not list):
            raise ValueError('Invalid capture or outcome')
        ack = CaptureAck(**dict(capture, routes=tuple(capture['routes'])))
        delivery = Delivery(_packet(payload['input']), ack, payload['tick'], tuple(payload['outcome']))
        return {'committed': store.apply(delivery, outcome_tag=tag)}
    raise ValueError('Unknown operation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database')
    options = parser.parse_args()
    try:
        keys = RouteOriginKeys(
            bytes.fromhex(os.environ['SARA_ROUTE_INPUT_KEY_HEX']),
            bytes.fromhex(os.environ['SARA_ROUTE_OUTCOME_KEY_HEX']),
        )
        raw = sys.stdin.buffer.read(MAX_MESSAGE_BYTES + 1)
        if len(raw) > MAX_MESSAGE_BYTES:
            raise ValueError('Message exceeds byte limit')
        message = json.loads(raw, object_pairs_hook=_unique_object)
        store = RouteReceiptStore(options.database, origin_keys=keys)
        if not Path(store.path).is_file():
            raise ValueError('Expected a staged database')
        result = dispatch(store, message)
    except (ValueError, TypeError, KeyError, OSError, RecursionError, sqlite3.Error):
        print('Route message rejected', file=sys.stderr)
        return 1
    print(json.dumps(result, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
