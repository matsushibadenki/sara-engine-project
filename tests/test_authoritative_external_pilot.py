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
    assert report["cycle_count"] == 2 and report["completed"]
    assert not report["promotion_ready"] and not report["automatic_default_enabled"]
    assert report["question_checks"] == [
        {"cycle": 0, "question_count": 1,
         "response_kind_counts": {"verified_answer": 1}},
        {"cycle": 1, "question_count": 1,
         "response_kind_counts": {"verified_answer": 1}},
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
