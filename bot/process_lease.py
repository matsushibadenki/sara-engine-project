"""Identify a metrics producer through a lock held for its process lifetime."""

import os
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_NB, LOCK_SH, flock
from tempfile import NamedTemporaryFile
from uuid import uuid4

LEASE_FILENAME = "metrics-producer.lock"


def valid_pid(value: object) -> bool:
    return type(value) is int and value > 0


def valid_launch_id(value: object) -> bool:
    return type(value) is str and len(value) == 32 and all(c in "0123456789abcdef" for c in value)


@contextmanager
def producer_lease(path: str):
    """Publish a new locked inode so readers never see a locked old token."""
    token = uuid4().hex
    with NamedTemporaryFile(mode="w", encoding="ascii", dir=os.path.dirname(path),
                            prefix=".metrics-producer-", delete=False) as lease:
        temporary_path = lease.name
        try:
            flock(lease, LOCK_EX)
            lease.write(f"{os.getpid()} {token}")
            lease.flush()
            os.replace(temporary_path, path)
            temporary_path = None
            yield token
        finally:
            if temporary_path is not None:
                os.unlink(temporary_path)


def live_launch_id(path: str, pid: int) -> str | None:
    """Read the launch ID only from an active lease belonging to the given PID."""
    try:
        with open(path, encoding="ascii") as lease:
            try:
                flock(lease, LOCK_SH | LOCK_NB)
            except BlockingIOError:
                owner, separator, token = lease.read(64).partition(" ")
                if owner == str(pid) and separator and valid_launch_id(token):
                    return token
    except (OSError, UnicodeError):
        pass
    return None
