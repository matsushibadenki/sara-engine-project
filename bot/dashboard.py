#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from bot.dashboard_pages import PAGES, render_page
from bot.status import build_status
PID_PATH = os.path.join(ROOT, "workspace", "autobot", "dashboard.pid")



class Handler(BaseHTTPRequestHandler):
    def _write_json(self, payload: dict, status_code: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _write_html(self, html: str, status_code: int = 200) -> None:
        raw = html.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, _format: str, *_args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        path = "/" if url.path == "/index.html" else url.path
        if path != "/api/status" and path not in PAGES:
            self.send_response(404)
            self.end_headers()
            return

        try:
            status = build_status(verbose=False)
        except Exception:
            logging.getLogger(__name__).exception("Autobot status collection failed")
            status = None

        status_code = 503 if status is None else 200
        if path == "/api/status":
            payload = {"error": "status_unavailable"} if status is None else status
            self._write_json(payload, status_code)
        else:
            lang = parse_qs(url.query).get("lang", ["en"])[0]
            self._write_html(render_page(path, lang, status), status_code)



def main() -> int:
    host = os.environ.get("AUTOBOT_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("AUTOBOT_DASHBOARD_PORT", "8765"))
    os.makedirs(os.path.dirname(PID_PATH), exist_ok=True)
    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"dashboard listening on http://{host}:{port}")
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        try:
            os.remove(PID_PATH)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
