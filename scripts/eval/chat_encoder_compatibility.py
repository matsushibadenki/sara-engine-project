"""Diagnose actual chat encoding without constructing a production agent."""

import json
import platform
import sys
import time
import tracemalloc
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.memory.sdr import SDREncoder
from sara_engine.utils.tokenizer import SaraTokenizer
from sara_engine.evaluation.verified_retrieval_compatibility import build_components, select_answer
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_parent_directory


class IsolatedChatEncoder:
    def __init__(self):
        # Use actual runtime methods with fresh vocabulary, no model I/O.
        self.agent = object.__new__(SaraAgent)
        self.agent.encoder = SDREncoder(2048, density=0.02, use_tokenizer=False, apply_vsa=False)
        self.agent.encoder.tokenizer = SaraTokenizer(load_existing=False)
        self.agent.encoder.use_tokenizer = True
        self.max_width = 0

    def __call__(self, text):
        if len(text) > 256:
            raise ValueError("Encoder input exceeds limit")
        self.agent._register_dynamic_vocab(text)
        result = self.agent.encoder.encode(text)
        self.max_width = max(self.max_width, len(result))
        return result


def evaluate(protocol, *, signature_width=64):
    rows = []
    for language in protocol["cases"]:
        for query in language["queries"]:
            encoder = IsolatedChatEncoder()
            record = {
                "entry_id":"door:r1", "topic":language["topic"], "text":language["text"],
                "language":language["language"], "source_ref":"fixture:door", "source_revision":"r1",
                "time_segment":1, "expires_at":None,
            }
            started = time.process_time_ns()
            cache, answers, legacy, _ = build_components([record], encoder=encoder, signature_width=signature_width)
            answer = select_answer(cache, answers, query["text"], language["language"], encoder=encoder)
            candidate_ms = (time.process_time_ns() - started) / 1e6
            started = time.process_time_ns()
            hits = legacy.search(encoder(query["text"]), top_k=1)
            legacy_ms = (time.process_time_ns() - started) / 1e6
            legacy_text = hits[0]["content"] if hits else ""
            def correct(text):
                return text == language["text"] if query["answerable"] else text == ""
            rows.append({
                "language":language["language"], "kind":query["kind"], "query":query["text"],
                "answerable":query["answerable"], "verified":answer, "legacy_text":legacy_text,
                "verified_correct":correct(answer["text"]), "legacy_correct":correct(legacy_text),
                "encoder_max_signature_width":encoder.max_width,
                "cache_signature_width":len(next(iter(cache.entries.values())).signature),
                "candidate_build_and_query_cpu_ms":candidate_ms,
                "legacy_warm_query_cpu_ms":legacy_ms,
            })
    return {
        "schema":protocol["schema"], "scope":protocol["scope"], "cases":rows,
        "signature_width_limit":signature_width,
        "verified_correct":sum(row["verified_correct"] for row in rows),
        "legacy_correct":sum(row["legacy_correct"] for row in rows),
        "case_count":len(rows), "all_cases_passed":all(row["verified_correct"] for row in rows),
        "normal_chat_promotion":False,
        "cost_caveat":"Candidate includes build+query; legacy is warm query. These timings cannot establish comparative speed or energy.",
        "encoder_configuration":"Fresh tokenizer; actual SaraAgent dynamic vocabulary and SDREncoder.encode; no learned semantic corpus or chat history",
    }


def main():
    raw = Path(processed_data_path("benchmark_fixtures", "chat_encoder_compatibility.json")).read_bytes()
    tracemalloc.start()
    report = evaluate(json.loads(raw))
    report["width_only_diagnostic"] = evaluate(json.loads(raw), signature_width=2048)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report.update(fixture_sha256=sha256(raw).hexdigest(), python=platform.python_version(),
                  platform=platform.platform(), harness_peak_traced_bytes=peak)
    path = Path(ensure_parent_directory(workspace_path("evaluation", "chat_encoder_compatibility.json")))
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("case_count", "verified_correct", "legacy_correct", "all_cases_passed", "harness_peak_traced_bytes")}))


if __name__ == "__main__":
    main()
