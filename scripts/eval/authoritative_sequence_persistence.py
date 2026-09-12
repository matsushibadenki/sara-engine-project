"""Evaluate restart-safe authoritative sequence persistence."""

import json
import sys
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from authoritative_evidence_v2 import NOW, FixtureTransport, configuration, make_payload
from sara_engine.memory.authoritative_evidence import (
    AuthoritativeEvidenceConfig, AuthoritativeEvidenceRuntime,
)
from sara_engine.memory.authoritative_sequence import AuthoritativeSequenceError
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_parent_directory


def _remove_state(path):
    for candidate in (path, Path(str(path) + ".lock")):
        candidate.unlink(missing_ok=True)


def _config(protocol, question_protocol, state_path):
    base = configuration(protocol, "en", question_protocol)
    return AuthoritativeEvidenceConfig(**{
        **base.__dict__, "sequence_state_path": str(state_path),
    })


def evaluate(protocol, v2_protocol, evidence_protocol, question_protocol, state_path):
    state_path = Path(state_path)
    _remove_state(state_path)
    rows = []
    config = _config(v2_protocol, question_protocol, state_path)
    initial = protocol["initial_sequence"]
    newer = protocol["next_sequence"]
    try:
        transport = FixtureTransport()
        transport.payloads = make_payload(
            v2_protocol, evidence_protocol, "en", sequence=initial,
            issued_offset=0, expiry_offset=120,
        )
        first = AuthoritativeEvidenceRuntime(config, clock=lambda: NOW, client_factory=transport)
        result = first.refresh()
        rows.append({
            "case": "cold_start_publish", "decision": result.decision,
            "correct": result.decision == "published" and first.accepted_sequence == initial,
        })

        replay_transport = FixtureTransport()
        replay_transport.payloads = transport.payloads
        replay = AuthoritativeEvidenceRuntime(
            config, clock=lambda: NOW, client_factory=replay_transport,
        )
        result = replay.refresh()
        rows.append({
            "case": "restart_replay", "decision": result.decision,
            "correct": result.decision == "publisher_rollback"
            and replay.accepted_sequence == initial,
        })

        newer_transport = FixtureTransport()
        newer_transport.payloads = make_payload(
            v2_protocol, evidence_protocol, "en", sequence=newer,
            issued_offset=0, expiry_offset=120,
        )
        restarted = AuthoritativeEvidenceRuntime(
            config, clock=lambda: NOW, client_factory=newer_transport,
        )
        result = restarted.refresh()
        rows.append({
            "case": "restart_newer", "decision": result.decision,
            "correct": result.decision == "published" and restarted.accepted_sequence == newer,
        })

        invalid_states = {
            "corrupt_state": b"not-json",
            "wrong_publisher_state": json.dumps({
                "schema": "sara-authoritative-sequence-v1", "publisher_id": "other",
                "scope": protocol["scope"], "accepted_sequence": newer,
            }).encode("utf-8"),
            "wrong_scope_state": json.dumps({
                "schema": "sara-authoritative-sequence-v1",
                "publisher_id": protocol["publisher_id"], "scope": "other",
                "accepted_sequence": newer,
            }).encode("utf-8"),
        }
        for name, body in invalid_states.items():
            state_path.write_bytes(body)
            decision = None
            try:
                AuthoritativeEvidenceRuntime(
                    config, clock=lambda: NOW, client_factory=FixtureTransport(),
                )
            except AuthoritativeSequenceError as exc:
                decision = exc.decision
            rows.append({
                "case": name, "decision": decision,
                "correct": decision == "watermark_invalid",
            })
    finally:
        _remove_state(state_path)
    expected = {name: decision for name, decision in protocol["cases"]}
    for row in rows:
        row["correct"] = row["correct"] and row["decision"] == expected[row["case"]]
    return {
        "schema": protocol["schema"], "cases": rows,
        "correct": sum(row["correct"] for row in rows), "case_count": len(rows),
        "passed": len(rows) == len(expected) and all(row["correct"] for row in rows),
        "automatic_default_enabled": False,
    }


def main():
    paths = {
        "protocol": Path(processed_data_path(
            "benchmark_fixtures", "authoritative_sequence_persistence_v1.json",
        )),
        "v2": Path(processed_data_path(
            "benchmark_fixtures", "authoritative_evidence_v2.json",
        )),
        "evidence": Path(processed_data_path(
            "benchmark_fixtures", "structured_query_contract_v1.json",
        )),
        "questions": Path(processed_data_path(
            "benchmark_fixtures", "verified_chat_routing_v1.json",
        )),
    }
    raw = {name: path.read_bytes() for name, path in paths.items()}
    state_path = Path(ensure_parent_directory(workspace_path(
        "evaluation", "authoritative_sequence_state.json",
    )))
    report = evaluate(
        *(json.loads(raw[name]) for name in ("protocol", "v2", "evidence", "questions")),
        state_path,
    )
    report["fixture_sha256"] = {
        name: sha256(body).hexdigest() for name, body in raw.items()
    }
    output = Path(ensure_parent_directory(workspace_path(
        "evaluation", "authoritative_sequence_persistence.json",
    )))
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        key: report[key] for key in (
            "correct", "case_count", "passed", "automatic_default_enabled",
        )
    }))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
