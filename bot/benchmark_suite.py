#!/usr/bin/env python3
"""Lightweight benchmark suite for autobot production model."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from bot.io_utils import atomic_write_json
from sara_engine.models.spiking_llm import SpikingLLM  # noqa: E402
from sara_engine.utils.project_paths import model_path, workspace_path, ensure_parent_directory  # noqa: E402


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    latency_ms: float
    score: float
    response: str
    reasons: list[str]


def _load_cases(path: str) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("benchmark cases must be a JSON array")
    return [x for x in raw if isinstance(x, dict)]


def _eval_case(llm: SpikingLLM, case: dict[str, object], args: argparse.Namespace) -> CaseResult:
    case_id = str(case.get("id", "unknown"))
    prompt = str(case.get("prompt", ""))
    must_contain_any = [str(x) for x in case.get("must_contain_any", []) if str(x).strip()] if isinstance(case.get("must_contain_any", []), list) else []
    must_not_contain_any = [str(x) for x in case.get("must_not_contain_any", []) if str(x).strip()] if isinstance(case.get("must_not_contain_any", []), list) else []
    max_chars = int(case.get("max_chars", args.max_chars_default))

    start = time.perf_counter()
    response = llm.generate(
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        stop_conditions=["\n"],
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    text = str(response).strip()

    passed = True
    reasons: list[str] = []
    score = 1.0

    if len(text) > max_chars:
        passed = False
        reasons.append(f"too_long:{len(text)}>{max_chars}")
        score -= 0.3

    if must_contain_any and not any(tok in text for tok in must_contain_any):
        passed = False
        reasons.append("missing_required_token")
        score -= 0.4

    if any(tok.lower() in text.lower() for tok in must_not_contain_any):
        passed = False
        reasons.append("contains_forbidden_token")
        score -= 0.5

    if len(text) < 2:
        passed = False
        reasons.append("empty_response")
        score -= 0.5

    if latency_ms > float(args.latency_warn_ms):
        reasons.append(f"latency_warn:{latency_ms:.1f}ms")
        score -= 0.1

    score = max(0.0, min(1.0, round(score, 4)))
    return CaseResult(
        case_id=case_id,
        passed=passed,
        latency_ms=round(latency_ms, 3),
        score=score,
        response=text,
        reasons=reasons,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark suite for autobot production model")
    parser.add_argument("--cases", default=os.path.join("bot", "benchmark_cases.json"))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--max-chars-default", type=int, default=280)
    parser.add_argument("--latency-warn-ms", type=float, default=3000.0)
    parser.add_argument("--tag", default="raw", help="Benchmark run tag, e.g. raw or hybrid")
    args = parser.parse_args()

    production_dir = model_path("autobot_self_organized", "production")
    weights_file = os.path.join(production_dir, "spiking_llm_weights.json")
    if not os.path.exists(weights_file):
        print(f"[ERROR] Production model not found: {weights_file}")
        return 1

    cases = _load_cases(args.cases)
    llm = SpikingLLM.from_pretrained(production_dir)

    results: list[CaseResult] = []
    for case in cases:
        results.append(_eval_case(llm, case, args))

    passed_count = sum(1 for r in results if r.passed)
    avg_latency = statistics.mean([r.latency_ms for r in results]) if results else 0.0
    avg_score = statistics.mean([r.score for r in results]) if results else 0.0

    recent_render_pairs = 0
    render_pairs = workspace_path("autobot", "render_pairs.jsonl")
    if os.path.exists(render_pairs):
        try:
            with open(render_pairs, "r", encoding="utf-8", errors="ignore") as f:
                recent_render_pairs = len([ln for ln in f.readlines()[-200:] if ln.strip()])
        except Exception:
            recent_render_pairs = 0

    payload = {
        "ts": datetime.utcnow().isoformat(),
        "tag": str(args.tag).strip().lower() or "raw",
        "model_dir": production_dir,
        "cases_total": len(results),
        "cases_passed": passed_count,
        "pass_rate": round((passed_count / max(1, len(results))), 4),
        "avg_latency_ms": round(float(avg_latency), 3),
        "avg_score": round(float(avg_score), 4),
        "recent_render_pairs": int(recent_render_pairs),
        "results": [
            {
                "id": r.case_id,
                "passed": r.passed,
                "latency_ms": r.latency_ms,
                "score": r.score,
                "reasons": r.reasons,
                "response": r.response,
            }
            for r in results
        ],
    }

    latest = workspace_path("autobot", "benchmark_latest.json")
    history = workspace_path("autobot", "benchmark_history.jsonl")
    ensure_parent_directory(latest)
    ensure_parent_directory(history)
    atomic_write_json(latest, payload)
    with open(history, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(
        f"[BENCH] pass_rate={payload['pass_rate']} "
        f"avg_score={payload['avg_score']} avg_latency_ms={payload['avg_latency_ms']} "
        f"cases={payload['cases_passed']}/{payload['cases_total']}"
    )

    return 0 if payload["pass_rate"] >= 0.8 else 2


if __name__ == "__main__":
    raise SystemExit(main())
