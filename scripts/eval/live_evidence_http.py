"""Exercise real loopback HTTP and page publication with an empty fixture scope."""

import json
import sys
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sara_engine.memory.evidence_http import EvidenceHTTPClient
from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.memory.evidence_wire import decode_evidence_page
from sara_engine.memory.topic_evidence_pages import EvidencePage, refresh_from_pages
from sara_engine.memory.topic_evidence_store import TopicEvidenceStore
from sara_engine.memory.structured_query import TopicQuery
from sara_engine.utils.project_paths import workspace_path, processed_data_path, ensure_parent_directory
from structured_query_contract import build_evidence


class Handler(BaseHTTPRequestHandler):
    evidence_payloads = {}
    def do_GET(self):
        path = urlsplit(self.path).path
        if path in self.evidence_payloads:
            body = json.dumps(self.evidence_payloads[path], ensure_ascii=False).encode("utf-8")
        elif path == "/invalid":
            body = b'{"scope":'
        elif path == "/oversized":
            body = b"x" * 1024
        else:
            body = json.dumps({"scope": "local", "snapshot": "v1", "index": 0,
                               "total_pages": 1, "total_records": 0}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def decode_empty_fixture(payload):
    if set(payload) != {"scope", "snapshot", "index", "total_pages", "total_records"} or payload["total_records"] != 0:
        raise ValueError("Unexpected fixture schema")
    return EvidencePage(records=(), **payload)


def main():
    protocol = json.loads(Path(processed_data_path("benchmark_fixtures", "structured_query_contract_v1.json")).read_text())
    questions = json.loads(Path(processed_data_path("benchmark_fixtures", "topic_question_v1.json")).read_text())
    agent = SaraAgent(input_size=256, hidden_size=256, compartments=["general"])
    for row in protocol["languages"]:
        Handler.evidence_payloads["/" + row["language"]] = {
            "schema": "sara-evidence-page-v1", "scope": "local", "snapshot": "v1",
            "index": 0, "total_pages": 1, "total_records": 2, "valid_until_segment": 8,
            "records": [asdict(record) for record in build_evidence(row, protocol["topics"])],
        }
    # Loopback only, ephemeral port; never expose a server on external interfaces.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    rows = []
    try:
        for path, expected in (("/valid", "published"), ("/invalid", "page_unavailable"), ("/oversized", "page_unavailable")):
            store = TopicEvidenceStore()
            store.publish((), expected_generation=0, now_segment=1)
            client = EvidenceHTTPClient(f"http://127.0.0.1:{server.server_port}{path}",
                                        decode_empty_fixture, max_bytes=512, timeout_seconds=2)
            result = refresh_from_pages(store, client, scope="local", expected_generation=1, now_segment=2)
            state = store.answer(TopicQuery(("missing",)), now_segment=2).result.decision
            expected_state = "incomplete_coverage" if expected == "published" else "evidence_unavailable"
            rows.append({"path": path, "decision": result.decision, "state": state,
                         "passed": result.decision == expected and state == expected_state and result.generation == 2})
        for language in protocol["languages"]:
            store = TopicEvidenceStore()
            client = EvidenceHTTPClient(f"http://127.0.0.1:{server.server_port}/{language['language']}",
                                        lambda payload: decode_evidence_page(payload, now_segment=3))
            result = refresh_from_pages(store, client, scope="local", expected_generation=0, now_segment=3)
            query_row = next(row for row in questions["languages"] if row["language"] == language["language"])
            question, requested, _ = query_row["cases"][1]
            answer = agent.answer_verified_question(question, evidence_store=store, language=language["language"],
                                                     aliases=tuple(map(tuple, query_row["aliases"])), now_segment=3)
            texts = [item.evidence[0].text for item in answer.result.items]
            rows.append({"language": language["language"], "decision": result.decision,
                         "passed": result.decision == "published" and texts == [language["texts"][protocol["topics"].index(topic)] for topic in requested]})
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
    report = {"scope": "Real loopback HTTP with initialized SaraAgent explicit API and multilingual fixtures; no TLS or external source",
              "cases": rows, "passed": all(row["passed"] for row in rows), "normal_chat_promotion": False}
    output = Path(ensure_parent_directory(workspace_path("evaluation", "live_evidence_http.json")))
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
