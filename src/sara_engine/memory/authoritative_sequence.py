"""Crash-safe monotonic sequence state for one authoritative publisher scope."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
from threading import RLock

from sara_engine.utils.project_paths import ensure_allowed_output_path, ensure_parent_directory


class AuthoritativeSequenceError(ValueError):
    def __init__(self, decision):
        self.decision = decision
        super().__init__(decision)


class AuthoritativeSequenceStore:
    """Persist one monotonic watermark with atomic replacement and file locking."""

    SCHEMA = "sara-authoritative-sequence-v1"
    MAX_STATE_BYTES = 1024

    def __init__(self, path, *, publisher_id, scope):
        if not isinstance(path, str) or not path:
            raise AuthoritativeSequenceError("watermark_invalid")
        for value in (publisher_id, scope):
            if not isinstance(value, str) or not 0 < len(value) <= 128 or value != value.strip():
                raise AuthoritativeSequenceError("watermark_invalid")
        try:
            self.path = Path(ensure_parent_directory(path))
            self.lock_path = Path(ensure_allowed_output_path(str(self.path) + ".lock"))
        except (OSError, TypeError, ValueError):
            raise AuthoritativeSequenceError("watermark_invalid") from None
        self.publisher_id = publisher_id
        self.scope = scope
        self._thread_lock = RLock()

    @contextmanager
    def _locked_file(self):
        descriptor = None
        try:
            descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            os.chmod(self.lock_path, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except AuthoritativeSequenceError:
            raise
        except OSError:
            raise AuthoritativeSequenceError("watermark_unavailable") from None
        finally:
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)

    @staticmethod
    def _unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate state key")
            value[key] = item
        return value

    def _read_locked(self):
        try:
            with self.path.open("rb") as handle:
                raw = handle.read(self.MAX_STATE_BYTES + 1)
        except FileNotFoundError:
            return None
        except OSError:
            raise AuthoritativeSequenceError("watermark_unavailable") from None
        if len(raw) > self.MAX_STATE_BYTES:
            raise AuthoritativeSequenceError("watermark_invalid")
        try:
            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=self._unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise AuthoritativeSequenceError("watermark_invalid") from None
        if not isinstance(payload, dict) or set(payload) != {
            "schema", "publisher_id", "scope", "accepted_sequence",
        }:
            raise AuthoritativeSequenceError("watermark_invalid")
        if (
            payload["schema"] != self.SCHEMA
            or payload["publisher_id"] != self.publisher_id
            or payload["scope"] != self.scope
            or type(payload["accepted_sequence"]) is not int
            or not 0 <= payload["accepted_sequence"] <= 2**63 - 1
        ):
            raise AuthoritativeSequenceError("watermark_invalid")
        return payload["accepted_sequence"]

    def load(self):
        with self._thread_lock, self._locked_file():
            return self._read_locked()

    def _write_locked(self, sequence):
        payload = {
            "schema": self.SCHEMA,
            "publisher_id": self.publisher_id,
            "scope": self.scope,
            "accepted_sequence": sequence,
        }
        body = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        descriptor = None
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix="." + self.path.name + ".", suffix=".tmp", dir=self.path.parent,
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
            parent = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        except OSError:
            raise AuthoritativeSequenceError("watermark_unavailable") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def advance(self, sequence):
        if type(sequence) is not int or not 0 <= sequence <= 2**63 - 1:
            raise AuthoritativeSequenceError("watermark_invalid")
        with self._thread_lock, self._locked_file():
            current = self._read_locked()
            if current is not None and sequence <= current:
                raise AuthoritativeSequenceError("publisher_rollback")
            self._write_locked(sequence)
            return sequence
