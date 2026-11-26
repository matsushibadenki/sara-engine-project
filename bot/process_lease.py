"""Identify a metrics producer through a lock held for its process lifetime."""

import os
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_NB, LOCK_SH, flock
from tempfile import NamedTemporaryFile
from uuid import uuid4

LEASE_FILENAME = "metrics-producer.lock"


@contextmanager
def producer_lease(path: str):
    """Publish a new locked inode so readers never see a locked old token."""
    token = uuid4().hex
    with NamedTemporaryFile(mode="w", encoding="ascii", dir=os.path.dirname(path),
                            prefix=".metrics-producer-", delete=False) as lease:
        temporary_path = lease.name
        try:
            flock(lease, LOCK_EX)
            lease.write(token)
            lease.flush()
            os.replace(temporary_path, path)
            temporary_path = None
            yield token
        finally:
            if temporary_path is not None:
                os.unlink(temporary_path)


def matches_live_producer(path: str, token: object) -> bool:
    if not isinstance(token, str) or not token:
        return False
    try:
        with open(path, encoding="ascii") as lease:
            try:
                flock(lease, LOCK_SH | LOCK_NB)
            except BlockingIOError:
                return lease.read(33) == token
    except (OSError, UnicodeError):
        return False
    return False
