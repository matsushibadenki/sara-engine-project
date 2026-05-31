#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from bot.status import build_status
PID_PATH = os.path.join(ROOT, "workspace", "autobot", "dashboard.pid")

_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>SARA Autobot Dashboard</title>
  <style>
    :root { --bg:#0f172a; --card:#111827; --text:#e5e7eb; --muted:#9ca3af; --ok:#10b981; --warn:#f59e0b; --bad:#ef4444; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; background:linear-gradient(120deg,#0b1020,#111827); color:var(--text); }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 20px; }
    h1 { margin: 0 0 16px; font-size: 22px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); gap:12px; }
    .card { background: rgba(17,24,39,0.9); border:1px solid #1f2937; border-radius: 12px; padding: 14px; }
    .k { color: var(--muted); font-size: 12px; }
    .v { font-size: 20px; margin-top: 4px; }
    .ok { color: var(--ok); } .warn { color: var(--warn); } .bad { color: var(--bad); }
    pre { white-space: pre-wrap; word-break: break-word; background:#0b1220; border-radius:10px; padding:10px; border:1px solid #1f2937; }
  </style>
</head>
<body>
<div class=\"wrap\">
  <h1>SARA Autobot Dashboard</h1>
  <div class=\"grid\" id=\"cards\"></div>
  <div class=\"card\" style=\"margin-top:12px\">
    <div class=\"k\">Raw JSON (latest status)</div>
    <pre id=\"raw\">loading...</pre>
  </div>
</div>
<script>
async function load() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const running = !!(d.process && d.process.running);
    const m = (d.metrics && d.metrics.data) || {};
    const logs = d.logs || {};
    const cards = [
      ['Process', running ? 'running' : 'stopped', running ? 'ok' : 'bad'],
      ['Queue Pending', String(m.queue_pending ?? 'n/a'), ''],
      ['New Samples', String(m.new_samples ?? 'n/a'), ''],
      ['Failed Items', String(m.failed_item_count ?? 'n/a'), (m.failed_item_count||0) > 0 ? 'warn' : 'ok'],
      ['Dead Letters', String(logs.dead_letter_lines ?? 'n/a'), (logs.dead_letter_lines||0) > 0 ? 'warn' : 'ok'],
      ['Critical Alerts (window)', String(logs.critical_alerts_recent ?? 'n/a'), (logs.critical_alerts_recent||0) > 0 ? 'warn' : 'ok'],
      ['Last Metrics TS', String(m.ts ?? 'n/a'), ''],
      ['Actions', (m.control_actions || []).join(',') || 'n/a', '']
    ];
    const el = document.getElementById('cards');
    el.innerHTML = cards.map(([k,v,c]) => `<div class=\"card\"><div class=\"k\">${k}</div><div class=\"v ${c}\">${v}</div></div>`).join('');
    document.getElementById('raw').textContent = JSON.stringify(d, null, 2);
  } catch (e) {
    document.getElementById('raw').textContent = 'dashboard fetch error: ' + e;
  }
}
load();
setInterval(load, 3000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _write_json(self, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _write_html(self, html: str) -> None:
        raw = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, _format: str, *_args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/status":
            self._write_json(build_status(verbose=False))
            return
        if path == "/" or path == "/index.html":
            self._write_html(_HTML)
            return
        self.send_response(404)
        self.end_headers()


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
