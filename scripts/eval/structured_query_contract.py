"""Evaluate a frozen explicit-topic contract without natural-language parsing."""

import json
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sara_engine.evaluation.verified_retrieval_compatibility import build_components
from sara_engine.memory.structured_query import TopicEvidence, TopicQuery, resolve_topic_query
from sara_engine.memory.verification_receipt import issue_verification_receipt
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_parent_directory


def bind_fixture_topic(topic_id, entry, answer):
    payload = {"schema": "sara-topic-answer-binding-v1", "topic_id": topic_id, "answer": answer.evidence_payload()}
    receipt = issue_verification_receipt(
        verifier_id="fixture-topic-verifier", verifier_version="v1",
        decision="verified_topic_answer_binding", evidence=payload,
        source_refs=(answer.source_ref,), source_revision=answer.source_revision,
        observed=True, source_backed=True, verified=True,
    )
    return TopicEvidence(topic_id, entry, answer, receipt)


def build_evidence(language, topics):
    records = [
        {"entry_id": f"topic:{index}", "topic": topic, "text": language["texts"][index],
         "language": language["language"], "source_ref": f"fixture:sensor:{index}",
         "source_revision": "r1", "time_segment": 1, "expires_at": None}
        for index, topic in enumerate(topics)
    ]
    cache, answers, _, _ = build_components(records)
    return tuple(bind_fixture_topic(topic, cache.entries[f"topic:{index}"], answers[f"topic:{index}"])
                 for index, topic in enumerate(topics))


def evaluate(protocol):
    rows = []
    for language in protocol["languages"]:
        evidence = build_evidence(language, protocol["topics"])
        for case in protocol["cases"]:
            query = TopicQuery(tuple(case["requested"]), tuple(case["excluded"]), language["language"], case.get("unresolved", False))
            result = resolve_topic_query(query, tuple(evidence[index] for index in case["available"]), now_segment=3)
            expected_topics = tuple(case["requested"]) if case["expected"] == "answer" else ()
            correct = result.decision == case["expected"] and tuple(item.topic_id for item in result.items) == expected_topics
            for item in result.items:
                index = protocol["topics"].index(item.topic_id)
                correct = correct and len(item.evidence) == 1 and item.evidence[0].text == language["texts"][index]
                correct = correct and item.evidence[0].source_ref == f"fixture:sensor:{index}" and item.evidence[0].source_revision == "r1"
            rows.append({"language": language["language"], "case": case["id"], "result": asdict(result), "correct": correct})
    return {"schema": protocol["schema"], "scope": protocol["scope"], "cases": rows,
            "correct": sum(row["correct"] for row in rows), "case_count": len(rows),
            "contract_passed": all(row["correct"] for row in rows), "normal_chat_promotion": False}


def main():
    raw = Path(processed_data_path("benchmark_fixtures", "structured_query_contract_v1.json")).read_bytes()
    report = evaluate(json.loads(raw))
    report["fixture_sha256"] = sha256(raw).hexdigest()
    output = Path(ensure_parent_directory(workspace_path("evaluation", "structured_query_contract_v1.json")))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("correct", "case_count", "contract_passed", "normal_chat_promotion")}))


if __name__ == "__main__":
    main()
