"""Shared-key provenance for route input and outcome producers."""
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import hmac

from sara_engine.evaluation.route_capture_layout import canonical


@dataclass(frozen=True, slots=True)
class RouteOriginKeys:
    input_key: bytes = field(repr=False)
    outcome_key: bytes = field(repr=False)

    def __post_init__(self):
        keys = (self.input_key, self.outcome_key)
        if any(type(key) is not bytes or len(key) < 32 for key in keys) or keys[0] == keys[1]:
            raise ValueError('Expected distinct input and outcome keys of at least 32 bytes')

    def binding(self):
        return canonical([sha256(key).hexdigest() for key in (self.input_key, self.outcome_key)])


def input_tag(key, packet, revision, digest):
    message = canonical([asdict(packet), revision, digest])
    return hmac.digest(key, b'sara-route-input-v1\0' + message, 'sha256')


def outcome_tag(key, delivery):
    return hmac.digest(key, b'sara-route-outcome-v1\0' + canonical(asdict(delivery)), 'sha256')


def verify_tag(expected, supplied):
    if type(supplied) is not bytes or not hmac.compare_digest(expected, supplied):
        raise ValueError('External origin verification failed')
