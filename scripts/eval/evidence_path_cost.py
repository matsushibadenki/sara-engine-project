"""Measure local decode/publication/question costs; excludes HTTP and training."""

import json
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sara_engine.memory.evidence_wire import decode_evidence_page
from sara_engine.memory.topic_evidence_store import TopicEvidenceStore
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_parent_directory
from structured_query_contract import build_evidence


def run_path(records, raw, question, language, aliases, *, wire):
    store = TopicEvidenceStore()
    if wire:
        page = decode_evidence_page(json.loads(raw), now_segment=3)
        selected, deadline = page.records, page.valid_until_segment
    else:
        selected, deadline = records, 8
    published = store.publish(selected, expected_generation=0, now_segment=3, valid_until_segment=deadline)
    if published.decision != "published":
        raise ValueError("Benchmark publication failed")
    return store.answer_question(question, language=language, aliases=aliases, now_segment=3)


def evaluate(questions, evidence_protocol, *, repeats=100):
    if type(repeats) is not int or not 1 <= repeats <= 1000:
        raise ValueError("Invalid benchmark repeat count")
    rows = []
    samples = {"direct": [], "wire": []}
    peaks = {"direct": [], "wire": []}
    for language in questions["languages"]:
        language_id = language["language"]
        source = next(row for row in evidence_protocol["languages"] if row["language"] == language_id)
        records = build_evidence(source, evidence_protocol["topics"])
        raw = json.dumps({"schema": "sara-evidence-page-v1", "scope": "local", "snapshot": "v1",
                          "index": 0, "total_pages": 1, "total_records": len(records), "valid_until_segment": 8,
                          "records": [asdict(record) for record in records]}, ensure_ascii=False).encode()
        aliases = tuple(tuple(pair) for pair in language["aliases"])
        for question, requested, excluded in language["cases"]:
            def run(arm):
                return run_path(records, raw, question, language_id, aliases, wire=arm == "wire")
            results = {arm: run(arm) for arm in samples}
            result = results["wire"].result
            if requested is None:
                correct = result.decision == "unsupported_question" and not result.items
            else:
                correct = result.decision == "answer" and [item.topic_id for item in result.items] == requested
                for item in result.items:
                    index = evidence_protocol["topics"].index(item.topic_id)
                    correct = correct and len(item.evidence) == 1
                    correct = correct and item.evidence[0].text == source["texts"][index]
                    correct = correct and item.evidence[0].source_ref == f"fixture:sensor:{index}"
                    correct = correct and item.evidence[0].source_revision == "r1"
            # Alternate order; time without tracemalloc instrumentation.
            for repeat in range(repeats):
                for arm in (("direct", "wire") if repeat % 2 == 0 else ("wire", "direct")):
                    start = time.process_time_ns()
                    repeated = run(arm)
                    samples[arm].append((time.process_time_ns() - start) / 1e6)
                    if repeated != results[arm]:
                        raise ValueError("Nondeterministic benchmark result")
            for arm in samples:
                tracemalloc.start()
                try:
                    run(arm)
                    peaks[arm].append(tracemalloc.get_traced_memory()[1])
                finally:
                    tracemalloc.stop()
            rows.append({"language": language_id, "question": question, "correct": bool(correct),
                         "equivalent": results["direct"] == results["wire"], "wire_bytes": len(raw)})
    metrics = {}
    for arm, values in samples.items():
        ordered = sorted(values)
        metrics[arm] = {"cpu_ms_median": statistics.median(values),
                        "cpu_ms_p95": ordered[max(0, (95 * len(values) + 99) // 100 - 1)],
                        "samples": len(values), "max_traced_peak_bytes": max(peaks[arm])}
    return {"scope": "Local JSON decode + verification + fresh store publication + restricted question; no HTTP",
            "cases": rows, "correct": sum(row["correct"] for row in rows),
            "equivalent": all(row["equivalent"] for row in rows), "metrics": metrics,
            "normal_chat_promotion": False,
            "limitations": "Synthetic development fixtures, two records per language; excludes network, fixture encoding/receipt issuance and interpreter imports. Allocation peaks exclude prebuilt inputs; CPU time is not latency or energy. No preregistered performance gate."}


def main():
    raw_questions = Path(processed_data_path("benchmark_fixtures", "topic_question_v1.json")).read_bytes()
    raw_evidence = Path(processed_data_path("benchmark_fixtures", "structured_query_contract_v1.json")).read_bytes()
    report = evaluate(json.loads(raw_questions), json.loads(raw_evidence))
    report.update(python=platform.python_version(), platform=platform.platform(),
                  question_sha256=sha256(raw_questions).hexdigest(), evidence_sha256=sha256(raw_evidence).hexdigest())
    output = Path(ensure_parent_directory(workspace_path("evaluation", "evidence_path_cost.json")))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("correct", "equivalent", "metrics")}))


if __name__ == "__main__":
    main()
