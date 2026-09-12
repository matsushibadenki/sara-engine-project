"""Evaluate the V2 authoritative publisher runtime with a bounded fake peer."""

import json
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.memory.authoritative_evidence import (
    AuthoritativeEvidenceConfig, AuthoritativeEvidenceRuntime,
)
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_parent_directory
from structured_query_contract import build_evidence


NOW = 2000000000


class FixtureTransport:
    def __init__(self):
        self.payloads = ()
        self.calls = []

    def __call__(self, endpoint, decoder, **options):
        transport = self
        class Client:
            def __call__(self, index):
                transport.calls.append(index)
                return decoder(transport.payloads[index])
        return Client()


def make_payload(protocol, evidence_protocol, language_id, *, sequence, issued_offset,
                 expiry_offset, publisher=None, pages=1):
    language = next(row for row in evidence_protocol["languages"] if row["language"] == language_id)
    records = build_evidence(language, evidence_protocol["topics"])
    chunks = tuple(records[index::pages] for index in range(pages))
    pages_payload = tuple({
        "schema": "sara-evidence-page-v2",
        "publisher_id": publisher or protocol["publisher_id"],
        "scope": protocol["scope"], "snapshot": f"sequence:{sequence}",
        "snapshot_sequence": sequence, "issued_at_epoch": NOW + issued_offset,
        "expires_at_epoch": NOW + expiry_offset, "index": index,
        "total_pages": pages, "total_records": len(records),
        "records": [asdict(record) for record in chunk],
    } for index, chunk in enumerate(chunks))
    # Match the HTTP client's JSON boundary: tuples become arrays and no Python
    # object identity passes from publisher fixtures into the decoder.
    return tuple(json.loads(json.dumps(page, ensure_ascii=False)) for page in pages_payload)


def configuration(protocol, language, question_protocol):
    row = next(item for item in question_protocol["languages"] if item["language"] == language)
    return AuthoritativeEvidenceConfig(
        endpoint="https://fixture.publisher.local/evidence",
        publisher_id=protocol["publisher_id"], scope=protocol["scope"],
        language=language, aliases=tuple(map(tuple, row["aliases"])),
        markers=tuple(row["markers"]), max_snapshot_lifetime_seconds=protocol["max_snapshot_lifetime_seconds"],
        max_future_skew_seconds=protocol["max_future_skew_seconds"],
    )


def evaluate(protocol, evidence_protocol, question_protocol):
    agent = SaraAgent(input_size=256, hidden_size=256, compartments=["general"])
    rows = []
    for language, question in protocol["languages"]:
        transport = FixtureTransport()
        transport.payloads = make_payload(protocol, evidence_protocol, language, sequence=1,
                                          issued_offset=0, expiry_offset=120, pages=2)
        runtime = AuthoritativeEvidenceRuntime(configuration(protocol, language, question_protocol),
                                               clock=lambda: NOW, client_factory=transport)
        refreshed = runtime.refresh()
        answer = runtime.chat(agent, question)
        source = next(item for item in evidence_protocol["languages"] if item["language"] == language)
        rows.append({"kind": "language", "language": language, "decision": refreshed.decision,
                     "correct": refreshed.decision == "published" and answer == source["texts"][1]
                     and runtime.status()["accepted_sequence"] == 1 and transport.calls == [0, 1]})

    language, question = protocol["languages"][0]
    transport = FixtureTransport()
    runtime = AuthoritativeEvidenceRuntime(configuration(protocol, language, question_protocol),
                                           clock=lambda: NOW, client_factory=transport)
    refusal = "I cannot answer this question from verified evidence."
    for name, sequence, issued_offset, expiry_offset, expected in protocol["scenarios"]:
        publisher = "wrong.publisher" if name == "wrong_publisher" else None
        transport.payloads = make_payload(
            protocol, evidence_protocol, language, sequence=sequence,
            issued_offset=issued_offset, expiry_offset=expiry_offset,
            publisher=publisher,
        )
        result = runtime.refresh()
        answer = runtime.chat(agent, question)
        expected_answer = expected == "published"
        rows.append({"kind": "lifecycle", "case": name, "decision": result.decision,
                     "correct": result.decision == expected and ((answer != refusal) == expected_answer)})
    return {"schema": protocol["schema"], "cases": rows,
            "correct": sum(row["correct"] for row in rows), "case_count": len(rows),
            "runtime_passed": all(row["correct"] for row in rows),
            "automatic_default_enabled": False}


def main():
    paths = {
        "protocol": Path(processed_data_path("benchmark_fixtures", "authoritative_evidence_v2.json")),
        "evidence": Path(processed_data_path("benchmark_fixtures", "structured_query_contract_v1.json")),
        "questions": Path(processed_data_path("benchmark_fixtures", "verified_chat_routing_v1.json")),
    }
    raw = {key: path.read_bytes() for key, path in paths.items()}
    report = evaluate(*(json.loads(raw[key]) for key in ("protocol", "evidence", "questions")))
    report["fixture_sha256"] = {key: sha256(value).hexdigest() for key, value in raw.items()}
    output = Path(ensure_parent_directory(workspace_path("evaluation", "authoritative_evidence_v2.json")))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("correct", "case_count", "runtime_passed", "automatic_default_enabled")}))


if __name__ == "__main__":
    main()
