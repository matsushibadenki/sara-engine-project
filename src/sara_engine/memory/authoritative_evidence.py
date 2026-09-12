"""Default-off runtime for one HTTPS evidence publisher and topic scope."""

from dataclasses import dataclass
import math
from threading import RLock
import time
from urllib.parse import urlsplit

from sara_engine.memory.evidence_http import EvidenceHTTPClient
from sara_engine.memory.evidence_wire import (
    AuthoritativeEvidenceError, decode_authoritative_evidence_page,
)
from sara_engine.memory.authoritative_sequence import (
    AuthoritativeSequenceError, AuthoritativeSequenceStore,
)
from sara_engine.memory.topic_evidence_pages import EvidencePageError, refresh_from_pages
from sara_engine.memory.topic_evidence_store import TopicEvidenceStore
from sara_engine.memory.verified_chat_router import route_verified_chat


@dataclass(frozen=True)
class AuthoritativeEvidenceConfig:
    endpoint: str
    publisher_id: str
    scope: str
    language: str
    aliases: tuple[tuple[str, str], ...]
    markers: tuple[str, ...] = ()
    ca_file: str | None = None
    timeout_seconds: float = 5.0
    max_bytes: int = 262144
    max_snapshot_lifetime_seconds: int = 3600
    max_future_skew_seconds: int = 30
    max_fetch_seconds: int = 8
    sequence_state_path: str | None = None


class AuthoritativeEvidenceRuntime:
    """Serialize refreshes and prevent publisher-sequence rollback.

    TLS authenticates the configured endpoint. Publisher identity, scope and
    time are assertions delivered through that channel. This does not prove the
    source's facts or completeness and performs no background refresh.
    """

    def __init__(self, config, *, clock=None, client_factory=EvidenceHTTPClient):
        if not isinstance(config, AuthoritativeEvidenceConfig):
            raise TypeError("AuthoritativeEvidenceConfig is required")
        url = urlsplit(config.endpoint) if isinstance(config.endpoint, str) else None
        if (
            url is None or not 0 < len(config.endpoint) <= 2048
            or url.scheme != "https" or not url.hostname
            or url.username or url.password or url.fragment
        ):
            raise ValueError("Authoritative evidence requires HTTPS")
        for value in (config.publisher_id, config.scope):
            if not isinstance(value, str) or not 0 < len(value) <= 128 or value != value.strip():
                raise ValueError("Invalid publisher configuration")
        if config.language not in ("en", "ja", "zh-CN"):
            raise ValueError("Invalid publisher language")
        probe = route_verified_chat(
            "sara configuration probe", language=config.language,
            aliases=config.aliases, markers=config.markers,
        )
        if probe.parse_decision == "invalid_config":
            raise ValueError("Invalid routing configuration")
        for value, maximum, name in (
            (config.max_snapshot_lifetime_seconds, 86400, "lifetime"),
            (config.max_future_skew_seconds, 86400, "clock skew"),
            (config.max_fetch_seconds, 60, "fetch time"),
        ):
            if type(value) is not int or not 0 < value <= maximum:
                raise ValueError("Invalid " + name)
        if (
            isinstance(config.timeout_seconds, bool)
            or not isinstance(config.timeout_seconds, (int, float))
            or not math.isfinite(config.timeout_seconds)
            or not 0 < config.timeout_seconds <= 60
            or type(config.max_bytes) is not int
            or not 1 <= config.max_bytes <= 1048576
            or (config.ca_file is not None
                and (not isinstance(config.ca_file, str) or not config.ca_file))
        ):
            raise ValueError("Invalid transport configuration")
        if (clock is not None and not callable(clock)) or not callable(client_factory):
            raise ValueError("Invalid runtime dependency")
        # Validate endpoint, trust material and transport limits before storing
        # the configuration. The client performs no request during construction.
        client_factory(config.endpoint, lambda value: value, timeout_seconds=config.timeout_seconds,
                       max_bytes=config.max_bytes, ca_file=config.ca_file)
        sequence_store = None
        accepted_sequence = None
        if config.sequence_state_path is not None:
            sequence_store = AuthoritativeSequenceStore(
                config.sequence_state_path, publisher_id=config.publisher_id,
                scope=config.scope,
            )
            accepted_sequence = sequence_store.load()
        self.config = config
        self.store = TopicEvidenceStore(max_age_segments=config.max_snapshot_lifetime_seconds)
        self._clock = clock or time.time
        self._client_factory = client_factory
        self._sequence_store = sequence_store
        self._accepted_sequence = accepted_sequence
        self._lock = RLock()

    def _now(self):
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Invalid publisher clock")
        value = int(value)
        if value < 0:
            raise ValueError("Invalid publisher clock")
        return value

    @property
    def accepted_sequence(self):
        with self._lock:
            return self._accepted_sequence

    def refresh(self):
        with self._lock:
            expected_generation = self.store.generation
            try:
                started = self._now()
            except Exception:
                revoked = self.store.refresh_failed(expected_generation=expected_generation)
                return type(revoked)("clock_unavailable", revoked.generation)
            try:
                client = self._client_factory(
                    self.config.endpoint, lambda value: value,
                    timeout_seconds=self.config.timeout_seconds, max_bytes=self.config.max_bytes,
                    ca_file=self.config.ca_file,
                )
            except Exception:
                revoked = self.store.refresh_failed(expected_generation=expected_generation)
                return type(revoked)("page_unavailable", revoked.generation)
            metadata = None

            def fetch(index):
                nonlocal metadata
                try:
                    decoded = decode_authoritative_evidence_page(
                        client(index), expected_publisher=self.config.publisher_id,
                        expected_scope=self.config.scope, now_epoch=self._now(),
                        max_future_skew_seconds=self.config.max_future_skew_seconds,
                        max_snapshot_lifetime_seconds=self.config.max_snapshot_lifetime_seconds,
                    )
                except AuthoritativeEvidenceError as exc:
                    raise EvidencePageError(exc.decision) from None
                current = (decoded.publisher_id, decoded.snapshot_sequence, decoded.issued_at_epoch)
                if self._accepted_sequence is not None and decoded.snapshot_sequence <= self._accepted_sequence:
                    raise EvidencePageError("publisher_rollback")
                if metadata is None:
                    metadata = current
                elif current != metadata:
                    raise EvidencePageError("snapshot_mismatch")
                return decoded.page

            def persist_sequence():
                if self._sequence_store is None:
                    return None
                try:
                    self._accepted_sequence = self._sequence_store.advance(metadata[1])
                except AuthoritativeSequenceError as exc:
                    return exc.decision
                return None

            result = refresh_from_pages(
                self.store, fetch, scope=self.config.scope,
                expected_generation=expected_generation, now_segment=started,
                clock=self._now, max_fetch_segments=self.config.max_fetch_seconds,
                before_publish=persist_sequence if self._sequence_store is not None else None,
            )
            if result.decision == "published" and self._sequence_store is None:
                self._accepted_sequence = metadata[1]
            return result

    def chat(self, agent, text):
        """Route one input through the configured default-off evidence scope."""
        return agent.chat(
            text, evidence_store=self.store, evidence_language=self.config.language,
            evidence_aliases=self.config.aliases, evidence_markers=self.config.markers,
            evidence_now_segment=self._now(), evidence_auto=True,
        )

    def status(self):
        """Return bounded non-secret state; omit endpoint, CA path and aliases."""
        with self._lock:
            return {
                "schema": "sara-authoritative-evidence-runtime-v2",
                "publisher_id": self.config.publisher_id,
                "scope": self.config.scope,
                "language": self.config.language,
                "accepted_sequence": self._accepted_sequence,
                "store_generation": self.store.generation,
                "sequence_persistence_enabled": self._sequence_store is not None,
                "automatic_default_enabled": False,
            }
