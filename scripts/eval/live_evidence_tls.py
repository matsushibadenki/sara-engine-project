"""Verify local TLS trust and hostname checks using a temporary test certificate."""

import json
import ssl
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sara_engine.memory.evidence_http import EvidenceHTTPClient
from sara_engine.utils.project_paths import workspace_path, ensure_output_directory, ensure_parent_directory


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"transport":"tls-fixture"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    scratch = ensure_output_directory(workspace_path("tests", "live-evidence-tls"))
    rows = []
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        key, cert = Path(directory) / "key.pem", Path(directory) / "cert.pem"
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                        "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost",
                        "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True, timeout=20)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for name, host, ca, should_pass in (
                ("trusted_hostname", "localhost", str(cert), True),
                ("untrusted_certificate", "localhost", None, False),
                ("hostname_mismatch", "127.0.0.1", str(cert), False),
            ):
                client = EvidenceHTTPClient(f"https://{host}:{server.server_port}/", lambda value: value,
                                            ca_file=ca, timeout_seconds=2)
                try:
                    value = client(0)
                    passed = should_pass and value == {"transport": "tls-fixture"}
                    decision = "received"
                except ssl.SSLCertVerificationError:
                    passed = not should_pass
                    decision = "certificate_rejected"
                rows.append({"case": name, "decision": decision, "passed": passed})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
    report = {"scope": "Local TLS transport only; ephemeral self-signed test certificate, no external publisher",
              "cases": rows, "passed": all(row["passed"] for row in rows)}
    output = Path(ensure_parent_directory(workspace_path("evaluation", "live_evidence_tls.json")))
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
