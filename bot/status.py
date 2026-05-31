#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read_json(path: str) -> tuple[dict[str, Any] | None, str | None]:
    if not os.path.exists(path):
        return None, "not_found"
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None, "invalid_object"
        return obj, None
    except Exception as exc:
        return None, f"parse_error:{exc}"


def _tail_lines(path: str, n: int) -> list[str]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-n:]
        return [ln.rstrip("\n") for ln in lines]
    except Exception:
        return []


def _critical_alerts_recent(path: str, window_minutes: int) -> int:
    if not os.path.exists(path):
        return 0
    cutoff = datetime.utcnow() - timedelta(minutes=max(1, window_minutes))
    count = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-3000:]
    except Exception:
        return 0
    for line in reversed(lines):
        if "severity=CRITICAL" not in line:
            continue
        idx = line.find("ts=")
        if idx < 0:
            continue
        ts_start = idx + 3
        ts_end = line.find(" ", ts_start)
        raw_ts = line[ts_start:] if ts_end < 0 else line[ts_start:ts_end]
        try:
            ts = datetime.fromisoformat(raw_ts)
        except Exception:
            continue
        if ts < cutoff:
            break
        count += 1
    return count


def build_status(verbose: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {}

    pid_path = os.path.join(ROOT, "workspace", "autobot", "bot.pid")
    proc = {"pid_file": os.path.exists(pid_path), "running": False, "pid": None}
    if os.path.exists(pid_path):
        try:
            with open(pid_path, "r", encoding="utf-8") as f:
                pid = int(f.read().strip())
            proc["pid"] = pid
            os.kill(pid, 0)
            proc["running"] = True
        except Exception:
            proc["running"] = False
    data["process"] = proc

    files = {
        "metrics": os.path.join(ROOT, "workspace", "autobot", "metrics.json"),
        "promotion": os.path.join(ROOT, "workspace", "autobot", "model_registry.json"),
        "shutdown": os.path.join(ROOT, "workspace", "autobot", "shutdown_status.json"),
        "weekly": os.path.join(ROOT, "workspace", "autobot", "weekly_digest.json"),
        "benchmark": os.path.join(ROOT, "workspace", "autobot", "benchmark_latest.json"),
        "benchmark_compare": os.path.join(ROOT, "workspace", "autobot", "benchmark_hybrid_compare.json"),
        "audit": os.path.join(ROOT, "workspace", "autobot", "audit_snapshot.json"),
        "state": os.path.join(ROOT, "workspace", "autobot", "state.json"),
        "config": os.path.join(ROOT, "bot", "config.example.json"),
    }

    for key, path in files.items():
        obj, err = _read_json(path)
        data[key] = {"data": obj, "error": err}

    alerts = os.path.join(ROOT, "workspace", "autobot", "alerts.log")
    dead = os.path.join(ROOT, "workspace", "autobot", "dead_letter.jsonl")
    events = os.path.join(ROOT, "workspace", "autobot", "events.jsonl")

    cfg_window = 20
    cfg_data = data.get("config", {}).get("data") if isinstance(data.get("config"), dict) else None
    if isinstance(cfg_data, dict):
        try:
            cfg_window = int(cfg_data.get("critical_alert_window_minutes", 20) or 20)
        except Exception:
            cfg_window = 20

    data["logs"] = {
        "alerts_exists": os.path.exists(alerts),
        "dead_letters_exists": os.path.exists(dead),
        "events_exists": os.path.exists(events),
        "dead_letter_lines": int(subprocess.getoutput(f"wc -l < '{dead}'").strip() or "0") if os.path.exists(dead) else 0,
        "critical_alerts_recent": _critical_alerts_recent(alerts, cfg_window),
        "critical_alert_window_minutes": cfg_window,
    }

    if verbose:
        data["tails"] = {
            "alerts": _tail_lines(alerts, 10),
            "events": _tail_lines(events, 10),
            "weekly_text": _tail_lines(os.path.join(ROOT, "workspace", "autobot", "weekly_digest.txt"), 20),
        }

    return data


def print_text(status: dict[str, Any], verbose: bool = False) -> None:
    p = status["process"]
    print("== Bot Process ==")
    if not p["pid_file"]:
        print("pid file not found")
    elif p["running"]:
        print(f"running pid={p['pid']}")
    else:
        print(f"not running (stale pid={p['pid']})")

    print("\n== Metrics ==")
    m = status["metrics"]
    if m["error"]:
        print(f"metrics {m['error']}")
    else:
        d = m["data"]
        print(f"ts={d.get('ts')}")
        print(f"new_samples={d.get('new_samples')} queue_pending={d.get('queue_pending')}")
        print(f"jp_ratio={d.get('jp_ratio')} en_ratio={d.get('en_ratio')}")
        print(f"actions={','.join(d.get('control_actions', []))}")
        print(f"alert_suppressed_total={d.get('alert_suppressed_total')}")

    print("\n== Promotion ==")
    pr = status["promotion"]
    if pr["error"]:
        print(f"promotion {pr['error']}")
    else:
        d = pr["data"]
        print(f"last_attempt_promoted={d.get('last_attempt_promoted')} reason={d.get('last_attempt_reason')}")

    print("\n== Weekly Digest ==")
    wk = status["weekly"]
    if wk["error"]:
        print(f"weekly {wk['error']}")
    else:
        d = wk["data"]
        print(f"updated_at={d.get('updated_at')} days={d.get('days_count')}")

    print("\n== Benchmark ==")
    bm = status["benchmark"]
    if bm["error"]:
        print(f"benchmark {bm['error']}")
    else:
        d = bm["data"]
        print(
            f"pass_rate={d.get('pass_rate')} avg_score={d.get('avg_score')} "
            f"avg_latency_ms={d.get('avg_latency_ms')} cases={d.get('cases_passed')}/{d.get('cases_total')} "
            f"tag={d.get('tag')} recent_render_pairs={d.get('recent_render_pairs')}"
        )

    print("\n== Hybrid Compare ==")
    bc = status["benchmark_compare"]
    if bc["error"]:
        print(f"hybrid_compare {bc['error']}")
    else:
        d = bc["data"]
        delta = d.get("delta", {}) if isinstance(d.get("delta"), dict) else {}
        print(
            f"raw_tag={d.get('raw_tag')} hybrid_tag={d.get('hybrid_tag')} "
            f"delta_pass_rate={delta.get('pass_rate')} delta_latency_ms={delta.get('avg_latency_ms')} "
            f"delta_score={delta.get('avg_score')}"
        )

    print("\n== Audit Snapshot ==")
    au = status["audit"]
    if au["error"]:
        print(f"audit {au['error']}")
    else:
        d = au["data"]
        print(f"ts={d.get('ts')} cycle={d.get('cycle')}")
        prom = d.get("promotion", {}) if isinstance(d.get("promotion"), dict) else {}
        print(f"last_attempt_promoted={prom.get('last_attempt_promoted')} eval_passed={prom.get('last_eval_passed')}")

    print("\n== Runtime State ==")
    st = status["state"]
    if st["error"]:
        print(f"state {st['error']}")
    else:
        d = st["data"]
        print(f"replay_cursor={d.get('replay_cursor')} last_eval_passed={d.get('last_eval_passed')}")

    print("\n== Config Baseline ==")
    cfg = status["config"]
    if cfg["error"]:
        print(f"config {cfg['error']}")
    else:
        c = cfg["data"]
        print(f"promotion_policy={c.get('promotion_policy')} compliance_preset={c.get('compliance_preset')}")
        print(f"semantic_hamming_threshold={c.get('semantic_hamming_threshold')} promotion_min_score={c.get('promotion_min_score')}")
        print(f"max_records_lines={c.get('max_records_lines')} max_corpus_lines={c.get('max_corpus_lines')}")
        print(f"critical_alert_window_minutes={c.get('critical_alert_window_minutes')} critical_alert_threshold={c.get('critical_alert_threshold')}")
        print(f"replay_interval_sec={c.get('replay_interval_sec')} replay_samples_per_cycle={c.get('replay_samples_per_cycle')} replay_min_quality={c.get('replay_min_quality')}")
        print(f"alert_dedup_window_sec={c.get('alert_dedup_window_sec')}")
        print(f"benchmark_min_pass_rate={c.get('benchmark_min_pass_rate')} benchmark_max_latency_ms={c.get('benchmark_max_latency_ms')}")

    print("\n== Alerts Summary ==")
    logs = status.get("logs", {})
    print(
        f"critical_alerts_recent={logs.get('critical_alerts_recent')} "
        f"window_minutes={logs.get('critical_alert_window_minutes')} "
        f"dead_letter_lines={logs.get('dead_letter_lines')}"
    )

    if verbose:
        print("\n== Tail Alerts ==")
        for ln in status.get("tails", {}).get("alerts", []):
            print(ln)


def main() -> int:
    parser = argparse.ArgumentParser(description="Show bot runtime status")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    status = build_status(verbose=args.verbose)
    if args.as_json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print_text(status, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
