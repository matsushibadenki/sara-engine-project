import importlib.util
import json
from pathlib import Path

import pytest

from sara_engine.memory.evidence_http import EvidenceNotModified
from sara_engine.utils.project_paths import workspace_path


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/eval/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PILOT = load("authoritative_external_pilot")
V2 = load("authoritative_evidence_v2")
PROTOCOL = json.loads((ROOT / "data/processed/benchmark_fixtures/authoritative_evidence_v2.json").read_text())
EVIDENCE = json.loads((ROOT / "data/processed/benchmark_fixtures/structured_query_contract_v1.json").read_text())
QUESTIONS = json.loads((ROOT / "data/processed/benchmark_fixtures/verified_chat_routing_v1.json").read_text())


class ScriptedTransport:
    def __init__(self, payloads):
        self.payloads = payloads

    def __call__(self, endpoint, decoder, **options):
        transport = self

        class Client:
            def __call__(self, index):
                if options["after_sequence"] == 1:
                    raise EvidenceNotModified()
                return decoder(transport.payloads[index])

        return Client()


@pytest.fixture
def pilot_payload(request):
    identity = request.node.name.replace("[", "-").replace("]", "-")
    state = Path(workspace_path("tests", "external-pilot", identity + ".json"))
    state.parent.mkdir(parents=True, exist_ok=True)
    for path in (state, Path(str(state) + ".lock")):
        path.unlink(missing_ok=True)
    base = V2.configuration(PROTOCOL, "en", QUESTIONS)
    question = next(row[1] for row in PROTOCOL["languages"] if row[0] == "en")
    payload = {
        "schema": PILOT.CONFIG_SCHEMA,
        "endpoint": base.endpoint, "publisher_id": base.publisher_id,
        "scope": base.scope, "language": base.language,
        "aliases": [list(row) for row in base.aliases],
        "markers": list(base.markers), "ca_file": base.ca_file,
        "timeout_seconds": base.timeout_seconds, "max_bytes": base.max_bytes,
        "max_snapshot_lifetime_seconds": base.max_snapshot_lifetime_seconds,
        "max_future_skew_seconds": base.max_future_skew_seconds,
        "max_fetch_seconds": base.max_fetch_seconds,
        "sequence_state_path": str(state), "questions": [question],
    }
    yield payload
    for path in (state, Path(str(state) + ".lock")):
        path.unlink(missing_ok=True)


def test_two_cycle_pilot_publishes_then_preserves_on_304(pilot_payload):
    payloads = V2.make_payload(
        PROTOCOL, EVIDENCE, "en", sequence=1, issued_offset=0, expiry_offset=120,
    )
    ticks = iter((1.0, 1.1, 2.0, 2.05))
    report = PILOT.run_pilot(
        pilot_payload, cycles=2, client_factory=ScriptedTransport(payloads),
        clock=lambda: V2.NOW, performance_clock=lambda: next(ticks),
    )
    assert report["decision_counts"] == {"published": 1, "publisher_not_modified": 1}
    assert [
        (refresh["accepted_sequence_before_refresh"], refresh["accepted_sequence"])
        for refresh in report["refreshes"]
    ] == [(None, 1), (1, 1)]
    assert report["cycle_count"] == 2 and report["completed"]
    assert not report["promotion_ready"] and not report["automatic_default_enabled"]
    assert report["question_checks"] == [
        {"cycle": 0, "question_count": 1,
         "response_kind_counts": {"verified_answer": 1},
         "response_kinds_in_question_order": ["verified_answer"]},
        {"cycle": 1, "question_count": 1,
         "response_kind_counts": {"verified_answer": 1},
         "response_kinds_in_question_order": ["verified_answer"]},
    ]
    rendered = repr(report)
    assert pilot_payload["endpoint"] not in rendered
    assert pilot_payload["sequence_state_path"] not in rendered
    assert pilot_payload["questions"][0] not in rendered


def test_example_configuration_is_parseable_and_remains_disabled():
    path = ROOT / "workspace/config/authoritative_external_pilot.example.json"
    payload = PILOT.load_configuration(path)
    config, questions = PILOT.parse_configuration(payload)
    assert config.endpoint == "https://publisher.example/evidence"
    assert config.sequence_state_path.startswith("data/interim/")
    assert questions == ("Report tank temperature.",)


@pytest.mark.parametrize("change", [
    {"questions": []}, {"questions": ["x" * 257]},
    {"questions": ["Tell me a short story about the moon."]},
    {"sequence_state_path": None}, {"unexpected": True},
])
def test_pilot_configuration_fails_closed(pilot_payload, change):
    payload = {**pilot_payload, **change}
    with pytest.raises(ValueError):
        PILOT.run_pilot(payload, cycles=1, client_factory=ScriptedTransport(()))


@pytest.mark.parametrize("cycles", [0, 33, True])
def test_pilot_cycle_bound(pilot_payload, cycles):
    with pytest.raises(ValueError):
        PILOT.run_pilot(pilot_payload, cycles=cycles, client_factory=ScriptedTransport(()))


@pytest.mark.parametrize("mode", [("--check-config",), ("--cycles", "1")])
@pytest.mark.parametrize("suffix", ["", ".lock"])
def test_cli_refuses_report_overlapping_sequence(pilot_payload, monkeypatch, mode, suffix):
    monkeypatch.setattr(PILOT, "load_configuration", lambda path: pilot_payload)

    def unexpected_run(*args, **kwargs):
        pytest.fail("Pilot started with an overlapping report path")

    monkeypatch.setattr(PILOT, "run_pilot", unexpected_run)
    output = pilot_payload["sequence_state_path"] + suffix
    with pytest.raises(ValueError, match="report path overlaps"):
        PILOT.main(["--config", "pilot.json", "--output", output, *mode])
    assert not Path(output).exists()


def test_failed_report_serialization_preserves_previous_report(pilot_payload, monkeypatch):
    sequence_path = Path(pilot_payload["sequence_state_path"])
    output = sequence_path.with_suffix(".report.json")
    output.write_text('{"previous": true}\n', encoding="utf-8")
    monkeypatch.setattr(PILOT, "load_configuration", lambda path: pilot_payload)
    monkeypatch.setattr(PILOT, "run_pilot", lambda *args, **kwargs: {"invalid": object()})
    try:
        existing_files = set(output.parent.iterdir())
        with pytest.raises(TypeError, match="not JSON serializable"):
            PILOT.main(["--config", "pilot.json", "--output", str(output), "--cycles", "1"])
        assert output.read_text(encoding="utf-8") == '{"previous": true}\n'
        assert set(output.parent.iterdir()) == existing_files
    finally:
        output.unlink()


def test_pilot_revision_lifecycle_across_runtime_restart(pilot_payload):
    def snapshot(sequence):
        return V2.make_payload(
            PROTOCOL, EVIDENCE, "en", sequence=sequence,
            issued_offset=0, expiry_offset=120,
        )

    first = PILOT.run_pilot(
        pilot_payload, cycles=2,
        client_factory=ScriptedTransport(snapshot(1)),
        clock=lambda: V2.NOW,
    )
    assert first["decision_counts"] == {"published": 1, "publisher_not_modified": 1}

    responses = iter([(1, snapshot(1)), (1, snapshot(2))])

    def restarted_transport(endpoint, decoder, *, after_sequence, **options):
        # The runtime constructs an unused client during configuration validation.
        def fetch(index):
            expected_validator, pages = next(responses)
            assert after_sequence == expected_validator
            return decoder(pages[index])

        return fetch

    restarted = PILOT.run_pilot(
        pilot_payload, cycles=2, client_factory=restarted_transport,
        clock=lambda: V2.NOW,
    )
    assert restarted["decision_counts"] == {"publisher_rollback": 1, "published": 1}
    assert [
        (refresh["decision"], refresh["accepted_sequence_before_refresh"],
         refresh["accepted_sequence"])
        for refresh in restarted["refreshes"]
    ] == [("publisher_rollback", 1, 1), ("published", 1, 2)]
    assert next(responses, None) is None
    assert not restarted["promotion_ready"] and not restarted["automatic_default_enabled"]


def test_https_pilot_lifecycle_across_cli_processes(pilot_payload):
    import ssl
    import subprocess
    import sys
    import tempfile
    import time
    from threading import Thread

    publisher = load("live_authoritative_evidence")
    staged_pages = []

    class StagedPublisher(publisher.Handler):
        def do_GET(self):
            if staged_pages:
                publisher.set_payload(staged_pages.pop(0))
            super().do_GET()

    scratch = Path(pilot_payload["sequence_state_path"]).parent
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        root = Path(directory)
        key, cert = root / "key.pem", root / "cert.pem"
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost",
            "-keyout", str(key), "-out", str(cert),
        ], check=True, capture_output=True, timeout=20)
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(cert, key)
        with publisher.ThreadingHTTPServer(("127.0.0.1", 0), StagedPublisher) as server:
            server.socket = tls.wrap_socket(server.socket, server_side=True)
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                state, config = root / "sequence.json", root / "config.json"
                payload = {
                    **pilot_payload, "endpoint": f"https://localhost:{server.server_port}/evidence",
                    "ca_file": str(cert), "sequence_state_path": str(state),
                }
                offset = int(time.time()) - V2.NOW
                for index, (offered, validator, before, after, decision) in enumerate([
                    (2, None, None, 2, "published"),
                    (3, None, 2, 3, "published"),
                    (3, 3, 3, 3, "publisher_not_modified"),
                    (2, None, 3, 3, "publisher_rollback"),
                    (4, None, 3, 4, "published"),
                    (5, None, 4, 5, "published"),
                    (6, None, 5, 6, "published"),
                ]):
                    pages = V2.make_payload(
                        PROTOCOL, EVIDENCE, "en", sequence=offered,
                        issued_offset=offset, expiry_offset=offset + 120,
                    )
                    page = pages[0]
                    questions = payload["questions"]
                    if offered >= 5:
                        # The complete revision retains the vent and omits the tank.
                        page["records"] = page["records"][:1]
                        page["total_records"] = 1
                    if offered in (4, 6):
                        questions = [*questions, "What is west vent state?"] if offered == 4 else [
                            "What is west vent state?",
                        ]
                    config.write_text(json.dumps({**payload, "questions": questions}), encoding="utf-8")
                    publisher.set_payload(page, not_modified_sequence=validator)
                    output = root / f"report-{index}.json"
                    result = subprocess.run([
                        sys.executable, str(ROOT / "scripts/eval/authoritative_external_pilot.py"),
                        "--config", str(config), "--output", str(output), "--cycles", "1",
                    ], capture_output=True, text=True, timeout=20)
                    assert result.returncode == 0, result.stderr
                    report = json.loads(output.read_text(encoding="utf-8"))
                    refresh, = report["refreshes"]
                    assert refresh["decision"] == decision
                    assert refresh["accepted_sequence_before_refresh"] == before
                    assert refresh["accepted_sequence"] == after
                    assert json.loads(state.read_text(encoding="utf-8"))["accepted_sequence"] == after
                    if offered >= 4:
                        expected_kinds = {
                            4: {"verified_answer": 2},
                            5: {"verified_abstention": 1},
                            6: {"verified_answer": 1},
                        }
                        assert report["question_checks"][0]["response_kind_counts"] == expected_kinds[offered]
                    assert not report["promotion_ready"] and not report["automatic_default_enabled"]
                for sequence in (7, 8):
                    page, = V2.make_payload(
                        PROTOCOL, EVIDENCE, "en", sequence=sequence,
                        issued_offset=offset, expiry_offset=offset + 120,
                    )
                    if sequence == 8:
                        page["records"] = page["records"][:1]
                        page["total_records"] = 1
                    staged_pages.append(page)
                config.write_text(json.dumps({
                    **payload, "questions": [*payload["questions"], "What is west vent state?"],
                }), encoding="utf-8")
                output = root / "same-process-withdrawal.json"
                result = subprocess.run([
                    sys.executable, str(ROOT / "scripts/eval/authoritative_external_pilot.py"),
                    "--config", str(config), "--output", str(output), "--cycles", "2",
                ], capture_output=True, text=True, timeout=20)
                assert result.returncode == 0, result.stderr
                report = json.loads(output.read_text(encoding="utf-8"))
                assert [
                    (row["decision"], row["accepted_sequence_before_refresh"], row["accepted_sequence"])
                    for row in report["refreshes"]
                ] == [("published", 6, 7), ("published", 7, 8)]
                assert [
                    row["response_kinds_in_question_order"] for row in report["question_checks"]
                ] == [
                    ["verified_answer", "verified_answer"],
                    ["verified_abstention", "verified_answer"],
                ]
                assert json.loads(state.read_text(encoding="utf-8"))["accepted_sequence"] == 8
            finally:
                server.shutdown()
                worker.join(timeout=5)
