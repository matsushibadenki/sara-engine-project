#!/usr/bin/env python3
"""Compare benchmark history between raw and hybrid (Chromium-assisted) runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from bot.io_utils import atomic_write_json
from sara_engine.utils.project_paths import workspace_path, ensure_parent_directory


def _load_jsonl(path: str) -> list[dict[str, object]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _avg(items: list[float]) -> float:
    if not items:
        return 0.0
    return float(sum(items) / max(1, len(items)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare raw vs hybrid benchmark runs")
    parser.add_argument("--history", default=workspace_path("autobot", "benchmark_history.jsonl"))
    parser.add_argument("--raw-tag", default="raw")
    parser.add_argument("--hybrid-tag", default="hybrid")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    rows = _load_jsonl(args.history)
    raw_rows = [r for r in rows if str(r.get("tag", "")).strip().lower() == args.raw_tag.lower()][-args.window :]
    hybrid_rows = [r for r in rows if str(r.get("tag", "")).strip().lower() == args.hybrid_tag.lower()][-args.window :]

    raw_pass = [float(r.get("pass_rate", 0.0) or 0.0) for r in raw_rows]
    raw_lat = [float(r.get("avg_latency_ms", 0.0) or 0.0) for r in raw_rows]
    raw_score = [float(r.get("avg_score", 0.0) or 0.0) for r in raw_rows]

    hy_pass = [float(r.get("pass_rate", 0.0) or 0.0) for r in hybrid_rows]
    hy_lat = [float(r.get("avg_latency_ms", 0.0) or 0.0) for r in hybrid_rows]
    hy_score = [float(r.get("avg_score", 0.0) or 0.0) for r in hybrid_rows]

    raw_avg_pass = _avg(raw_pass)
    hy_avg_pass = _avg(hy_pass)
    raw_avg_lat = _avg(raw_lat)
    hy_avg_lat = _avg(hy_lat)
    raw_avg_score = _avg(raw_score)
    hy_avg_score = _avg(hy_score)

    out = {
        "ts": datetime.utcnow().isoformat(),
        "window": int(args.window),
        "raw_tag": args.raw_tag,
        "hybrid_tag": args.hybrid_tag,
        "raw_count": len(raw_rows),
        "hybrid_count": len(hybrid_rows),
        "raw": {
            "avg_pass_rate": round(raw_avg_pass, 4),
            "avg_latency_ms": round(raw_avg_lat, 3),
            "avg_score": round(raw_avg_score, 4),
        },
        "hybrid": {
            "avg_pass_rate": round(hy_avg_pass, 4),
            "avg_latency_ms": round(hy_avg_lat, 3),
            "avg_score": round(hy_avg_score, 4),
        },
        "delta": {
            "pass_rate": round(hy_avg_pass - raw_avg_pass, 4),
            "avg_latency_ms": round(hy_avg_lat - raw_avg_lat, 3),
            "avg_score": round(hy_avg_score - raw_avg_score, 4),
        },
    }

    out_path = workspace_path("autobot", "benchmark_hybrid_compare.json")
    ensure_parent_directory(out_path)
    atomic_write_json(out_path, out)

    print(
        f"[HYBRID-COMPARE] raw_pass={out['raw']['avg_pass_rate']} "
        f"hybrid_pass={out['hybrid']['avg_pass_rate']} delta={out['delta']['pass_rate']} "
        f"latency_delta_ms={out['delta']['avg_latency_ms']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
