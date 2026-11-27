#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

if __package__:
    from bot.process_lease import LEASE_FILENAME, live_launch_id, valid_launch_id, valid_pid
else:
    from process_lease import LEASE_FILENAME, live_launch_id, valid_launch_id, valid_pid

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
            # Preserve slice semantics for zero and negative limits.
            lines = list(deque(f, maxlen=n)) if n > 0 else f.readlines()[-n:]
        return [ln.rstrip("\n") for ln in lines]
    except Exception:
        return []


def _count_lines(path: str) -> int | None:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0
    except OSError:
        return None


def _critical_alerts_recent(path: str, window_minutes: int) -> int | None:
    if not os.path.exists(path):
        return 0
    cutoff = datetime.utcnow() - timedelta(minutes=max(1, window_minutes))
    count = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = deque(f, maxlen=3000)
    except FileNotFoundError:
        return 0
    except OSError:
        return None
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
            if raw_ts.endswith("Z"):
                raw_ts = raw_ts[:-1] + "+00:00"
            ts = datetime.fromisoformat(raw_ts)
        except Exception:
            continue
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        if ts >= cutoff:
            count += 1
    return count


def build_status(verbose: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {}

    pid_path = os.path.join(ROOT, "workspace", "autobot", "bot.pid")
    lease = os.path.join(ROOT, "workspace", "autobot", LEASE_FILENAME)
    launch_id = None
    proc = {"pid_file": os.path.exists(pid_path), "running": False, "pid": None}
    if os.path.exists(pid_path):
        try:
            with open(pid_path, "r", encoding="utf-8") as f:
                pid = int(f.read().strip())
            proc["pid"] = pid
            if pid > 0:
                os.kill(pid, 0)
                launch_id = live_launch_id(lease, pid)
                proc["running"] = launch_id is not None
        except Exception:
            proc["running"] = False
    proc["launch_id"] = launch_id
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

    metrics_source = data["metrics"]
    metrics = metrics_source["data"] or {}
    if metrics_source["data"] is None:
        reason = "missing" if metrics_source["error"] == "not_found" else "unreadable"
    elif launch_id is None:
        reason = "no_live_producer"
    elif metrics.get("producer_pid") in (None, "") or metrics.get("producer_launch_id") in (None, ""):
        reason = "identity_missing"
    elif not valid_pid(metrics.get("producer_pid")) or not valid_launch_id(metrics.get("producer_launch_id")):
        reason = "identity_invalid"
    elif metrics.get("producer_pid") != proc["pid"]:
        reason = "pid_mismatch"
    elif metrics.get("producer_launch_id") != launch_id:
        reason = "launch_id_mismatch"
    else:
        reason = "verified_live_producer"
    metrics_current = reason == "verified_live_producer"
    metrics_source.update(current=metrics_current, reason=reason)
    runtime_window = metrics.get("critical_alert_window_minutes")
    if metrics_current and type(runtime_window) is int and runtime_window > 0:
        cfg_window = runtime_window
        window_source = "runtime"
    else:
        cfg = data["config"]["data"] or {}
        fallback = cfg.get("critical_alert_window_minutes")
        cfg_window = fallback if type(fallback) is int and fallback > 0 else 20
        window_source = "fallback"

    data["logs"] = {
        "alerts_exists": os.path.exists(alerts),
        "dead_letters_exists": os.path.exists(dead),
        "events_exists": os.path.exists(events),
        "dead_letter_lines": _count_lines(dead),
        "critical_alerts_recent": _critical_alerts_recent(alerts, cfg_window),
        "critical_alert_window_minutes": cfg_window,
        "critical_alert_window_source": window_source,
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
