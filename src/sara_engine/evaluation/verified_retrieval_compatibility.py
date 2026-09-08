"""Compare existing sparse retrieval components on isolated synthetic fixtures.

The shared character-bigram encoder is deliberately transparent. Human-authored
topic text is a declared prior; no labels, expected answers, or scenario names
enter the retrieval calls. This is not an end-to-end chat comparison.
"""

from dataclasses import asdict
from hashlib import sha256
from typing import Mapping
import json

from sara_engine.memory.event_state_cache import EventStateCandidate, VerifiedHierarchicalEventStateCache
from sara_engine.memory.ltm import SparseMemoryStore
from sara_engine.memory.verified_answer import VerifiedAnswer, resolve_verified_answer
from sara_engine.memory.verification_receipt import issue_verification_receipt


def encode(text: str) -> tuple[int, ...]:
    if not isinstance(text, str) or len(text) > 256:
        raise ValueError("Text input exceeds compatibility limit")
    normalized = "".join(char for char in text.casefold() if char.isalnum())
    grams = {normalized[index:index + 2] for index in range(len(normalized) - 1)}
    ids = {int.from_bytes(sha256(gram.encode()).digest()[:8], "big") for gram in grams}
    if len(ids) > 64:
        raise ValueError("Signature exceeds compatibility limit")
    return tuple(sorted(ids))


def _record(row, *, revision="r1", source="fixture:door:a", text=None, time=1, expiry=None):
    return {
        "entry_id": source + ":" + revision, "topic": row["evidence_topic"],
        "text": text or row["current"], "language": row["language"],
        "source_ref": source, "source_revision": revision,
        "time_segment": time, "expires_at": expiry,
    }


def build_inputs(row, scenario):
    first = _record(row)
    question = row["question"]
    records = [first]
    expected = {"text": first["text"], "source_ref": first["source_ref"], "source_revision": "r1"}
    if scenario == "missing":
        question = row["missing_question"]
        expected = None
    elif scenario.startswith("conflicting_sources"):
        records.append(_record(row, source="fixture:door:b", text=row["old"]))
        if scenario.endswith("reversed"):
            records.reverse()
        expected = None
    elif scenario.startswith("revision"):
        records = [_record(row, text=row["old"]), _record(row, revision="r2", time=2)]
        expected["source_revision"] = "r2"
        if scenario.endswith("reversed"):
            records.reverse()
    elif scenario == "expired":
        records = [_record(row, expiry=3)]
        expected = None
    elif scenario == "missing_binding":
        expected = None
    return records, question, expected


def build_components(records, *, omit_binding=False, encoder=encode, signature_width=64):
    if len(records) > 12:
        raise ValueError("Record count exceeds compatibility limit")
    cache = VerifiedHierarchicalEventStateCache(retention_profile="fixed", max_entries=12, top_k=12, max_signature_width=signature_width)
    # Initialize only the existing in-memory search state; never load/save a model.
    legacy = object.__new__(SparseMemoryStore)
    legacy.memories = []
    answers = {}
    admissions = []
    for record in records:
        signature = tuple(encoder(record["topic"]))
        candidate = EventStateCandidate.from_verified_evidence(
            verifier_id="fixture-state-verifier", evidence=record,
            entry_id=record["entry_id"], signature=signature,
            source_ref=record["source_ref"], source_revision=record["source_revision"],
            time_segment=record["time_segment"], expires_at=record["expires_at"],
            observed=True, source_backed=True, verified=True, resonance_score=0.95,
        )
        admission = cache.admit(candidate)
        admissions.append(admission.decision)
        stored = cache.entries.get(admission.entry_id)
        # The trusted fixture verifier binds text to the admitted stable ID.
        # An older observation must never overwrite a newer answer binding.
        binding_current = bool(
            admission.accepted and stored is not None
            and stored.source_ref == record["source_ref"]
            and stored.source_revision == record["source_revision"]
        )
        bound_id = admission.entry_id
        payload = {
            "schema": "sara-verified-answer-binding-v1",
            **{key: record[key] for key in ("entry_id", "text", "language", "source_ref", "source_revision")},
        }
        payload["entry_id"] = bound_id
        receipt = issue_verification_receipt(
            verifier_id="fixture-answer-verifier", verifier_version="v1",
            decision="verified_answer_binding", evidence=payload,
            source_refs=(record["source_ref"],), source_revision=record["source_revision"],
            observed=True, source_backed=True, verified=True,
        )
        if not omit_binding and binding_current:
            answers[bound_id] = VerifiedAnswer(
                bound_id, record["text"], record["language"],
                record["source_ref"], record["source_revision"], receipt,
            )
        legacy.memories.append({
            "sdr": list(signature), "content": record["text"], "timestamp": record["time_segment"],
            "type": "episodic", "metadata": {
                "source_ref": record["source_ref"], "source_revision": record["source_revision"],
            },
        })
    return cache, answers, legacy, admissions


def select_answer(cache, answers, question, language, *, encoder=encode):
    result = cache.retrieve(encoder(question), now_segment=3, top_k=12, signature_match="contained")
    resolved = []
    for match in result.matches:
        entry_id = match["entry_id"]
        answer = resolve_verified_answer(
            cache.entries.get(entry_id), answers.get(entry_id), now_segment=3, language=language,
        )
        if answer.decision != "answer":
            return {"decision": answer.decision, "text": "", "source_ref": "", "source_revision": ""}
        resolved.append(answer)
    if not resolved:
        return {"decision": "no_match", "text": "", "source_ref": "", "source_revision": ""}
    if len({answer.text for answer in resolved}) > 1:
        return {"decision": "conflicting_evidence", "text": "", "source_ref": "", "source_revision": ""}
    return asdict(resolved[0])


def evaluate(protocol: Mapping) -> dict:
    rows = []
    for language in protocol["languages"]:
        for scenario in protocol["scenarios"]:
            records, question, expected = build_inputs(language, scenario)
            cache, answers, legacy, admissions = build_components(records, omit_binding=scenario == "missing_binding")
            selected = select_answer(cache, answers, question, language["language"])
            hits = legacy.search(list(encode(question)), top_k=1)
            legacy_answer = {
                "text": hits[0]["content"] if hits else "",
                "source_ref": hits[0]["metadata"]["source_ref"] if hits else "",
                "source_revision": hits[0]["metadata"]["source_revision"] if hits else "",
            }
            def correct(answer):
                return not answer["text"] if expected is None else all(answer[key] == value for key, value in expected.items())
            rows.append({
                "language": language["language"], "scenario": scenario,
                "answerable": expected is not None, "expected": expected,
                "verified": selected, "legacy": legacy_answer,
                "verified_correct": correct(selected), "legacy_correct": correct(legacy_answer),
                "admissions": admissions, "retained_entries": len(cache.entries),
            })
    metrics = {}
    for arm in ("verified", "legacy"):
        answerable = [row for row in rows if row["answerable"]]
        abstain = [row for row in rows if not row["answerable"]]
        metrics[arm] = {
            "answerable_correct_rate": sum(row[arm + "_correct"] for row in answerable) / len(answerable),
            "required_abstention_rate": sum(row[arm + "_correct"] for row in abstain) / len(abstain),
            "answered_count": sum(bool(row[arm]["text"]) for row in rows),
            "case_count": len(rows),
        }
    return {
        "schema": "sara-verified-retrieval-comparison-v2", "scope": protocol["scope"],
        "interpretation": "Development regression after inspecting v1 failures; not untouched evaluation",
        "matching": "Verified: full evidence-signature containment; legacy: existing overlap ranking. Shared encoder.",
        "metrics": metrics, "cases": rows,
        "component_contract_passed": all(row["verified_correct"] for row in rows),
        "normal_chat_promotion": False,
        "limits": "12 fixture records; 256 encoder input characters; 64 signature IDs. No CPU/byte-budget claim.",
    }
