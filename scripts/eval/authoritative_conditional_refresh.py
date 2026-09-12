"""Evaluate bounded conditional refresh without extending evidence freshness."""

import json
import sys
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from authoritative_evidence_v2 import NOW, configuration, make_payload
from sara_engine.memory.authoritative_evidence import (
    AuthoritativeEvidenceConfig, AuthoritativeEvidenceRuntime,
)
from sara_engine.memory.evidence_http import EvidenceNotModified
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_parent_directory


class ConditionalTransport:
    def __init__(self):
        self.payloads = ()
        self.not_modified = False
        self.not_modified_index = 0
        self.options = []

    def __call__(self, endpoint, decoder, **options):
        transport = self
        transport.options.append(dict(options))

        class Client:
            def __call__(self, index):
                if transport.not_modified and index == transport.not_modified_index:
                    raise EvidenceNotModified()
                return decoder(transport.payloads[index])

        return Client()


def _remove(path):
    for candidate in (path, Path(str(path) + ".lock")):
        candidate.unlink(missing_ok=True)


def _config(v2_protocol, question_protocol, state_path=None):
    base = configuration(v2_protocol, "en", question_protocol)
    return AuthoritativeEvidenceConfig(**{
        **base.__dict__, "sequence_state_path": (
            None if state_path is None else str(state_path)
        ),
    })


def _answer_decision(runtime, question, now):
    return runtime.store.answer_question(
        question, language=runtime.config.language,
        aliases=runtime.config.aliases, now_segment=now,
    ).result.decision


def evaluate(protocol, v2_protocol, evidence_protocol, question_protocol, state_path):
    state_path = Path(state_path)
    expired_path = state_path.with_name(state_path.stem + "-expired.json")
    for path in (state_path, expired_path):
        _remove(path)
    sequence = protocol["sequence"]
    question = next(row[1] for row in v2_protocol["languages"] if row[0] == "en")
    rows = []
    try:
        transport = ConditionalTransport()
        transport.payloads = make_payload(
            v2_protocol, evidence_protocol, "en", sequence=sequence,
            issued_offset=0, expiry_offset=120,
        )
        config = _config(v2_protocol, question_protocol, state_path)
        runtime = AuthoritativeEvidenceRuntime(
            config, clock=lambda: NOW, client_factory=transport,
        )
        initial = runtime.refresh()
        initial_generation = runtime.store.generation
        rows.append({
            "case": "initial_200", "decision": initial.decision,
            "correct": initial.decision == "published"
            and _answer_decision(runtime, question, NOW) == "answer",
        })

        transport.not_modified = True
        unchanged = runtime.refresh()
        rows.append({
            "case": "unchanged_304", "decision": unchanged.decision,
            "correct": unchanged.decision == "publisher_not_modified"
            and runtime.store.generation == initial_generation
            and runtime.accepted_sequence == sequence
            and transport.options[-1]["after_sequence"] == sequence
            and _answer_decision(runtime, question, NOW) == "answer",
        })

        transport.not_modified = False
        replayed = runtime.refresh()
        rows.append({
            "case": "replayed_200", "decision": replayed.decision,
            "correct": replayed.decision == "publisher_rollback"
            and _answer_decision(runtime, question, NOW) == "evidence_unavailable",
        })

        cold_transport = ConditionalTransport()
        cold_transport.not_modified = True
        restarted = AuthoritativeEvidenceRuntime(
            config, clock=lambda: NOW, client_factory=cold_transport,
        )
        cold = restarted.refresh()
        rows.append({
            "case": "cold_restart_304", "decision": cold.decision,
            "correct": cold.decision == "publisher_not_modified"
            and restarted.store.generation == 0
            and _answer_decision(restarted, question, NOW) == "evidence_unavailable",
        })

        time_state = {"now": NOW}
        expiring_transport = ConditionalTransport()
        expiring_transport.payloads = make_payload(
            v2_protocol, evidence_protocol, "en", sequence=sequence,
            issued_offset=0, expiry_offset=120,
        )
        expiring = AuthoritativeEvidenceRuntime(
            _config(v2_protocol, question_protocol, expired_path),
            clock=lambda: time_state["now"], client_factory=expiring_transport,
        )
        assert expiring.refresh().decision == "published"
        expiring_transport.not_modified = True
        time_state["now"] = NOW + 120
        expired = expiring.refresh()
        rows.append({
            "case": "expired_local_304", "decision": expired.decision,
            "correct": expired.decision == "publisher_not_modified"
            and _answer_decision(expiring, question, NOW + 120) == "snapshot_expired",
        })
    finally:
        for path in (state_path, expired_path):
            _remove(path)
    expected = {name: decision for name, decision in protocol["cases"]}
    for row in rows:
        row["correct"] = row["correct"] and row["decision"] == expected[row["case"]]
    return {
        "schema": protocol["schema"], "cases": rows,
        "correct": sum(row["correct"] for row in rows), "case_count": len(rows),
        "passed": len(rows) == len(expected) and all(row["correct"] for row in rows),
        "freshness_extended_by_304": False,
        "automatic_default_enabled": False,
    }


def main():
    paths = {
        "protocol": Path(processed_data_path(
            "benchmark_fixtures", "authoritative_conditional_refresh_v1.json",
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
        "evaluation", "authoritative_conditional_sequence.json",
    )))
    report = evaluate(
        *(json.loads(raw[name]) for name in ("protocol", "v2", "evidence", "questions")),
        state_path,
    )
    report["fixture_sha256"] = {
        name: sha256(body).hexdigest() for name, body in raw.items()
    }
    output = Path(ensure_parent_directory(workspace_path(
        "evaluation", "authoritative_conditional_refresh.json",
    )))
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        key: report[key] for key in (
            "correct", "case_count", "passed", "freshness_extended_by_304",
            "automatic_default_enabled",
        )
    }))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
