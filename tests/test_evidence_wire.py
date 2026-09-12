import copy
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from sara_engine.memory.evidence_wire import (
    AuthoritativeEvidenceError, decode_authoritative_evidence_page,
    decode_evidence_page,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wire_fixture", ROOT / "scripts/eval/structured_query_contract.py")
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)
PROTOCOL = json.loads((ROOT / "data/processed/benchmark_fixtures/structured_query_contract_v1.json").read_text())


def payload(language):
    records = FIXTURE.build_evidence(language, PROTOCOL["topics"])
    return json.loads(json.dumps({"schema": "sara-evidence-page-v1", "scope": "local", "snapshot": "v1",
        "index": 0, "total_pages": 1, "total_records": 2, "valid_until_segment": 8,
        "records": [asdict(record) for record in records]}))


def authoritative_payload(language, *, now=2000000000):
    value = payload(language)
    value.update(schema="sara-evidence-page-v2", publisher_id="fixture.publisher.local",
                 snapshot_sequence=1, issued_at_epoch=now, expires_at_epoch=now + 120)
    value.pop("valid_until_segment")
    return value


@pytest.mark.parametrize("language", PROTOCOL["languages"])
def test_wire_roundtrip_preserves_received_bindings(language):
    original = payload(language)
    before = copy.deepcopy(original)
    page = decode_evidence_page(original, now_segment=3)
    assert [item.answer.text for item in page.records] == language["texts"]
    assert page.records[0].receipt.integrity_digest == original["records"][0]["receipt"]["integrity_digest"]
    assert original == before


@pytest.mark.parametrize("field,value", [("text", "tampered"), ("source_revision", "r2"), ("language", "fr")])
def test_tampered_answer_rejected(field, value):
    raw = payload(PROTOCOL["languages"][0])
    raw["records"][0]["answer"][field] = value
    with pytest.raises(ValueError):
        decode_evidence_page(raw, now_segment=3)


@pytest.mark.parametrize("mutation", ["flag", "signature", "extra", "topic", "expiry", "missing_receipt"])
def test_invalid_wire_is_not_coerced_or_repaired(mutation):
    raw = payload(PROTOCOL["languages"][0])
    row = raw["records"][0]
    if mutation == "flag":
        row["receipt"]["verified"] = "false"
    elif mutation == "signature":
        row["entry"]["signature"] = list(range(65))
    elif mutation == "extra":
        raw["trusted"] = True
    elif mutation == "topic":
        row["topic_id"] = "other"
    elif mutation == "expiry":
        raw["valid_until_segment"] = 3
    else:
        del row["answer"]["receipt"]
    with pytest.raises(ValueError):
        decode_evidence_page(raw, now_segment=3)


def test_authoritative_wire_preserves_v1_records():
    raw = authoritative_payload(PROTOCOL["languages"][0])
    decoded = decode_authoritative_evidence_page(
        raw, expected_publisher="fixture.publisher.local", expected_scope="local",
        now_epoch=2000000000, max_future_skew_seconds=5,
        max_snapshot_lifetime_seconds=300,
    )
    assert decoded.publisher_id == "fixture.publisher.local"
    assert decoded.snapshot_sequence == 1 and decoded.issued_at_epoch == 2000000000
    assert decoded.page.valid_until_segment == 2000000120
    assert decoded.page.records[0].answer.text == PROTOCOL["languages"][0]["texts"][0]


@pytest.mark.parametrize("mutation,decision", [
    ({"publisher_id": "other"}, "publisher_mismatch"),
    ({"scope": "other"}, "scope_mismatch"),
    ({"issued_at_epoch": 1999999880, "expires_at_epoch": 2000000000}, "snapshot_expired"),
    ({"issued_at_epoch": 2000000006}, "publisher_time_invalid"),
    ({"expires_at_epoch": 2000000301}, "publisher_time_invalid"),
    ({"snapshot_sequence": True}, "publisher_time_invalid"),
    ({"unexpected": 1}, "invalid_page"),
])
def test_authoritative_metadata_fails_closed(mutation, decision):
    raw = authoritative_payload(PROTOCOL["languages"][0])
    raw.update(mutation)
    with pytest.raises(AuthoritativeEvidenceError) as caught:
        decode_authoritative_evidence_page(
            raw, expected_publisher="fixture.publisher.local", expected_scope="local",
            now_epoch=2000000000, max_future_skew_seconds=5,
            max_snapshot_lifetime_seconds=300,
        )
    assert caught.value.decision == decision
