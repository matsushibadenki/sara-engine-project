import importlib.util
import json
import os
from pathlib import Path

import pytest

from sara_engine.memory.authoritative_evidence import (
    AuthoritativeEvidenceConfig, AuthoritativeEvidenceRuntime,
)
from sara_engine.memory.authoritative_sequence import (
    AuthoritativeSequenceError, AuthoritativeSequenceStore,
)
from sara_engine.utils.project_paths import workspace_path


ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/eval/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load("authoritative_evidence_v2")
PERSISTENCE_MODULE = load("authoritative_sequence_persistence")
PROTOCOL_RAW = (ROOT / "data/processed/benchmark_fixtures/authoritative_evidence_v2.json").read_bytes()
PROTOCOL = json.loads(PROTOCOL_RAW)
PERSISTENCE_RAW = (ROOT / "data/processed/benchmark_fixtures/authoritative_sequence_persistence_v1.json").read_bytes()
PERSISTENCE_PROTOCOL = json.loads(PERSISTENCE_RAW)
EVIDENCE = json.loads((ROOT / "data/processed/benchmark_fixtures/structured_query_contract_v1.json").read_text())
QUESTIONS = json.loads((ROOT / "data/processed/benchmark_fixtures/verified_chat_routing_v1.json").read_text())


def test_frozen_runtime_acceptance():
    from hashlib import sha256
    assert sha256(PROTOCOL_RAW).hexdigest() == "b496539cf6c3627d27499aaabf587ca08da2f2a6730aafa5e9a39414a863ed9b"
    report = MODULE.evaluate(PROTOCOL, EVIDENCE, QUESTIONS)
    assert report["correct"] == report["case_count"] == 10
    assert report["runtime_passed"] and not report["automatic_default_enabled"]


def test_status_omits_transport_and_catalog_details():
    transport = MODULE.FixtureTransport()
    config = MODULE.configuration(PROTOCOL, "en", QUESTIONS)
    runtime = AuthoritativeEvidenceRuntime(config, clock=lambda: MODULE.NOW, client_factory=transport)
    status = runtime.status()
    assert status == {"schema": "sara-authoritative-evidence-runtime-v2",
                      "publisher_id": PROTOCOL["publisher_id"], "scope": PROTOCOL["scope"],
                      "language": "en", "accepted_sequence": None, "store_generation": 0,
                      "sequence_persistence_enabled": False,
                      "automatic_default_enabled": False}
    assert config.endpoint not in repr(status) and repr(config.aliases) not in repr(status)


@pytest.mark.parametrize("change", [
    {"endpoint": "http://localhost/evidence"}, {"publisher_id": ""},
    {"endpoint": "https://placeholder@localhost/evidence"},
    {"endpoint": "https://localhost/evidence#fragment"},
    {"scope": " scope"}, {"language": "fr"}, {"aliases": ()},
    {"markers": ["sensor"]}, {"max_snapshot_lifetime_seconds": 0},
    {"max_future_skew_seconds": True}, {"max_fetch_seconds": 61},
    {"timeout_seconds": float("nan")}, {"max_bytes": True}, {"ca_file": ""},
])
def test_configuration_fails_before_any_request(change):
    transport = MODULE.FixtureTransport()
    values = MODULE.configuration(PROTOCOL, "en", QUESTIONS).__dict__ | change
    with pytest.raises((TypeError, ValueError, OSError)):
        AuthoritativeEvidenceRuntime(AuthoritativeEvidenceConfig(**values), clock=lambda: MODULE.NOW,
                                     client_factory=transport)
    assert transport.calls == []


def test_non_callable_clock_is_rejected_even_when_falsey():
    transport = MODULE.FixtureTransport()
    config = MODULE.configuration(PROTOCOL, "en", QUESTIONS)
    with pytest.raises(ValueError):
        AuthoritativeEvidenceRuntime(config, clock=0, client_factory=transport)


def test_clock_failure_revokes_current_store():
    transport = MODULE.FixtureTransport()
    config = MODULE.configuration(PROTOCOL, "en", QUESTIONS)
    ticks = iter((MODULE.NOW,) * 5 + (object(),))
    runtime = AuthoritativeEvidenceRuntime(config, clock=lambda: next(ticks), client_factory=transport)
    transport.payloads = MODULE.make_payload(PROTOCOL, EVIDENCE, "en", sequence=1,
                                             issued_offset=0, expiry_offset=120)
    assert runtime.refresh().decision == "published"
    result = runtime.refresh()
    assert result.decision == "clock_unavailable"
    assert runtime.store.generation == 2 and runtime.accepted_sequence == 1


@pytest.mark.parametrize("field,value", [
    ("publisher_id", "other.publisher"),
    ("snapshot_sequence", 2),
    ("issued_at_epoch", MODULE.NOW - 1),
    ("expires_at_epoch", MODULE.NOW + 121),
])
def test_multipage_publisher_metadata_mismatch_is_auditable(field, value):
    transport = MODULE.FixtureTransport()
    config = MODULE.configuration(PROTOCOL, "en", QUESTIONS)
    runtime = AuthoritativeEvidenceRuntime(config, clock=lambda: MODULE.NOW, client_factory=transport)
    payloads = list(MODULE.make_payload(
        PROTOCOL, EVIDENCE, "en", sequence=1, issued_offset=0,
        expiry_offset=120, pages=2,
    ))
    payloads[1][field] = value
    transport.payloads = tuple(payloads)
    result = runtime.refresh()
    expected = "publisher_mismatch" if field == "publisher_id" else "snapshot_mismatch"
    assert result.decision == expected
    assert runtime.store.generation == 1 and runtime.accepted_sequence is None


@pytest.fixture
def sequence_path(request):
    directory = Path(workspace_path("tests", "authoritative-sequence"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{request.node.name}.json"
    for candidate in (path, Path(str(path) + ".lock")):
        candidate.unlink(missing_ok=True)
    yield path
    for candidate in (path, Path(str(path) + ".lock")):
        candidate.unlink(missing_ok=True)


def test_sequence_store_is_atomic_private_and_monotonic(sequence_path):
    store = AuthoritativeSequenceStore(
        str(sequence_path), publisher_id=PROTOCOL["publisher_id"], scope=PROTOCOL["scope"],
    )
    assert store.load() is None
    assert store.advance(7) == 7
    assert store.load() == 7
    assert os.stat(sequence_path).st_mode & 0o777 == 0o600
    assert not tuple(sequence_path.parent.glob(f".{sequence_path.name}.*.tmp"))
    with pytest.raises(AuthoritativeSequenceError) as caught:
        store.advance(7)
    assert caught.value.decision == "publisher_rollback" and store.load() == 7


@pytest.mark.parametrize("payload", [
    b"not-json", b"{}", b'{"schema":"sara-authoritative-sequence-v1","publisher_id":"other","scope":"fixture.sensors","accepted_sequence":7}',
    b'{"schema":"sara-authoritative-sequence-v1","publisher_id":"fixture.publisher.local","scope":"other","accepted_sequence":7}',
    b'{"schema":"sara-authoritative-sequence-v1","publisher_id":"fixture.publisher.local","scope":"fixture.sensors","accepted_sequence":true}',
])
def test_sequence_store_rejects_corrupt_or_foreign_state(sequence_path, payload):
    sequence_path.write_bytes(payload)
    store = AuthoritativeSequenceStore(
        str(sequence_path), publisher_id=PROTOCOL["publisher_id"], scope=PROTOCOL["scope"],
    )
    with pytest.raises(AuthoritativeSequenceError) as caught:
        store.load()
    assert caught.value.decision == "watermark_invalid"


def test_runtime_rejects_replay_after_restart_and_accepts_newer(sequence_path):
    config = AuthoritativeEvidenceConfig(**{
        **MODULE.configuration(PROTOCOL, "en", QUESTIONS).__dict__,
        "sequence_state_path": str(sequence_path),
    })
    transport = MODULE.FixtureTransport()
    transport.payloads = MODULE.make_payload(
        PROTOCOL, EVIDENCE, "en", sequence=7, issued_offset=0, expiry_offset=120,
    )
    first = AuthoritativeEvidenceRuntime(config, clock=lambda: MODULE.NOW, client_factory=transport)
    assert first.refresh().decision == "published" and first.accepted_sequence == 7

    replay_transport = MODULE.FixtureTransport()
    replay_transport.payloads = transport.payloads
    restarted = AuthoritativeEvidenceRuntime(
        config, clock=lambda: MODULE.NOW, client_factory=replay_transport,
    )
    assert restarted.accepted_sequence == 7
    assert restarted.status()["sequence_persistence_enabled"] is True
    assert restarted.refresh().decision == "publisher_rollback"
    assert restarted.store.generation == 1 and restarted.accepted_sequence == 7

    newer_transport = MODULE.FixtureTransport()
    newer_transport.payloads = MODULE.make_payload(
        PROTOCOL, EVIDENCE, "en", sequence=8, issued_offset=0, expiry_offset=120,
    )
    recovered = AuthoritativeEvidenceRuntime(
        config, clock=lambda: MODULE.NOW, client_factory=newer_transport,
    )
    assert recovered.refresh().decision == "published"
    assert recovered.accepted_sequence == 8


def test_invalid_snapshot_does_not_advance_durable_sequence(sequence_path):
    config = AuthoritativeEvidenceConfig(**{
        **MODULE.configuration(PROTOCOL, "en", QUESTIONS).__dict__,
        "sequence_state_path": str(sequence_path),
    })
    transport = MODULE.FixtureTransport()
    payloads = list(MODULE.make_payload(
        PROTOCOL, EVIDENCE, "en", sequence=7, issued_offset=0,
        expiry_offset=120, pages=2,
    ))
    payloads[1]["snapshot_sequence"] = 8
    transport.payloads = tuple(payloads)
    runtime = AuthoritativeEvidenceRuntime(config, clock=lambda: MODULE.NOW, client_factory=transport)
    assert runtime.refresh().decision == "snapshot_mismatch"
    assert runtime.accepted_sequence is None and not sequence_path.exists()


def test_frozen_sequence_persistence_acceptance(sequence_path):
    from hashlib import sha256
    assert sha256(PERSISTENCE_RAW).hexdigest() == "924747e35341f6d2d8dfd83d9569ffc3c48d89ee4688aae3b0d240703a0a876b"
    report = PERSISTENCE_MODULE.evaluate(
        PERSISTENCE_PROTOCOL, PROTOCOL, EVIDENCE, QUESTIONS, sequence_path,
    )
    assert report["correct"] == report["case_count"] == 6
    assert report["passed"] and not report["automatic_default_enabled"]


def test_persistence_failure_revokes_without_exposing_snapshot(sequence_path, monkeypatch):
    config = AuthoritativeEvidenceConfig(**{
        **MODULE.configuration(PROTOCOL, "en", QUESTIONS).__dict__,
        "sequence_state_path": str(sequence_path),
    })
    transport = MODULE.FixtureTransport()
    transport.payloads = MODULE.make_payload(
        PROTOCOL, EVIDENCE, "en", sequence=7, issued_offset=0, expiry_offset=120,
    )
    runtime = AuthoritativeEvidenceRuntime(config, clock=lambda: MODULE.NOW, client_factory=transport)

    def fail(_):
        raise AuthoritativeSequenceError("watermark_unavailable")

    monkeypatch.setattr(runtime._sequence_store, "advance", fail)
    assert runtime.refresh().decision == "watermark_unavailable"
    assert runtime.accepted_sequence is None and runtime.store.generation == 1
