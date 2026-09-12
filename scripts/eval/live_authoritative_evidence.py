"""Run the V2 publisher runtime through real local TLS and initialized chat."""

import json
import ssl
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from authoritative_evidence_v2 import NOW, configuration, make_payload
from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.memory.authoritative_evidence import (
    AuthoritativeEvidenceConfig, AuthoritativeEvidenceRuntime,
)
from sara_engine.utils.project_paths import processed_data_path, workspace_path, ensure_output_directory, ensure_parent_directory


class Handler(BaseHTTPRequestHandler):
    payload = {}
    lock = Lock()

    def do_GET(self):
        with self.lock:
            body = json.dumps(self.payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def set_payload(payload):
    with Handler.lock:
        Handler.payload = payload


def main():
    protocol = json.loads(Path(processed_data_path("benchmark_fixtures", "authoritative_evidence_v2.json")).read_text())
    evidence = json.loads(Path(processed_data_path("benchmark_fixtures", "structured_query_contract_v1.json")).read_text())
    questions = json.loads(Path(processed_data_path("benchmark_fixtures", "verified_chat_routing_v1.json")).read_text())
    scratch = ensure_output_directory(workspace_path("tests", "live-authoritative-evidence"))
    rows = []
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        key, cert = Path(directory) / "key.pem", Path(directory) / "cert.pem"
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost",
            "-keyout", str(key), "-out", str(cert),
        ], check=True, capture_output=True, timeout=20)
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(cert, key)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.socket = tls.wrap_socket(server.socket, server_side=True)
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        agent = SaraAgent(input_size=256, hidden_size=256, compartments=["general"])
        try:
            runtimes = {}
            for language, question in protocol["languages"]:
                base = configuration(protocol, language, questions)
                config = AuthoritativeEvidenceConfig(**{
                    **base.__dict__, "endpoint": f"https://localhost:{server.server_port}/evidence",
                    "ca_file": str(cert),
                })
                runtime = AuthoritativeEvidenceRuntime(config, clock=lambda: NOW)
                set_payload(make_payload(protocol, evidence, language, sequence=1,
                                         issued_offset=0, expiry_offset=120)[0])
                refreshed = runtime.refresh()
                answer = runtime.chat(agent, question)
                source = next(row for row in evidence["languages"] if row["language"] == language)
                trace = agent.get_last_response_trace()
                rows.append({"language": language, "case": "initial",
                             "decision": refreshed.decision,
                             "passed": refreshed.decision == "published" and answer == source["texts"][1]
                             and trace["owners"] == ["verified_chat_router", "verified_topic_store"]})
                runtimes[language] = (runtime, question, source["texts"][1])

            runtime, question, expected = runtimes["en"]
            set_payload(make_payload(protocol, evidence, "en", sequence=1,
                                     issued_offset=0, expiry_offset=120)[0])
            replay = runtime.refresh()
            refused = runtime.chat(agent, question)
            rows.append({"language": "en", "case": "replay",
                         "decision": replay.decision,
                         "passed": replay.decision == "publisher_rollback" and refused != expected
                         and agent.get_last_response_trace()["kind"] == "verified_abstention"})

            set_payload(make_payload(protocol, evidence, "en", sequence=2,
                                     issued_offset=0, expiry_offset=120)[0])
            recovery = runtime.refresh()
            rows.append({"language": "en", "case": "recovery", "decision": recovery.decision,
                         "passed": recovery.decision == "published" and runtime.chat(agent, question) == expected
                         and runtime.accepted_sequence == 2})

            wrong = make_payload(protocol, evidence, "en", sequence=3, issued_offset=0,
                                 expiry_offset=120, publisher="wrong.publisher")[0]
            set_payload(wrong)
            mismatch = runtime.refresh()
            rows.append({"language": "en", "case": "publisher_mismatch", "decision": mismatch.decision,
                         "passed": mismatch.decision == "publisher_mismatch"
                         and runtime.chat(agent, question) != expected and runtime.accepted_sequence == 2})

            bad_config = AuthoritativeEvidenceConfig(**{
                **configuration(protocol, "en", questions).__dict__,
                "endpoint": f"https://localhost:{server.server_port}/evidence", "ca_file": None,
            })
            untrusted = AuthoritativeEvidenceRuntime(bad_config, clock=lambda: NOW)
            untrusted_result = untrusted.refresh()
            rows.append({"language": "en", "case": "untrusted_tls", "decision": untrusted_result.decision,
                         "passed": untrusted_result.decision == "page_unavailable"
                         and untrusted.store.generation == 1})
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)
    report = {
        "scope": "Real local TLS V2 publisher runtime; synthetic records and ephemeral certificate",
        "cases": rows, "passed": all(row["passed"] for row in rows),
        "automatic_default_enabled": False,
    }
    output = Path(ensure_parent_directory(workspace_path("evaluation", "live_authoritative_evidence.json")))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
