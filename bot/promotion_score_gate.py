from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreGateResult:
    passed: bool
    score: float
    reasons: list[str]


def evaluate_score_gate(
    *,
    eval_report: dict[str, object],
    snapshot: dict[str, object],
    min_score: float,
) -> ScoreGateResult:
    reasons: list[str] = []

    corpus_lines = int(eval_report.get("corpus_lines", 0) or 0)
    eval_passed = bool(eval_report.get("passed", False))
    queue_pending = int(snapshot.get("queue_pending", 0) or 0)
    failed_items = int(snapshot.get("failed_item_count", 0) or 0)
    dead_delta = int(snapshot.get("dead_letter_delta", 0) or 0)
    benchmark = eval_report.get("benchmark", {})
    benchmark_passed = True
    benchmark_pass_rate = 1.0
    benchmark_latency_ms = 0.0
    if isinstance(benchmark, dict) and bool(benchmark.get("available", False)):
        benchmark_passed = bool(benchmark.get("passed", False))
        benchmark_pass_rate = float(benchmark.get("pass_rate", 0.0) or 0.0)
        benchmark_latency_ms = float(benchmark.get("avg_latency_ms", 0.0) or 0.0)

    quality = 1.0 if eval_passed else 0.0
    quality *= min(1.0, corpus_lines / 2000.0)
    if benchmark:
        quality *= max(0.0, min(1.0, benchmark_pass_rate))

    stability = 1.0
    stability -= min(0.6, failed_items / 1000.0)
    stability -= min(0.3, dead_delta / 200.0)
    stability = max(0.0, stability)

    health = 1.0 - min(0.8, queue_pending / 5000.0)
    health = max(0.0, health)

    score = round(quality * 0.5 + stability * 0.3 + health * 0.2, 4)

    if not eval_passed:
        reasons.append("eval_report_failed")
    if not benchmark_passed:
        reasons.append("benchmark_failed")
    if benchmark_latency_ms > 0 and benchmark_latency_ms > 5000:
        reasons.append("benchmark_latency_high")
    if queue_pending > 2500:
        reasons.append("queue_pending_high")
    if failed_items > 300:
        reasons.append("failed_items_high")
    if dead_delta > 120:
        reasons.append("dead_letter_spike")

    passed = score >= float(min_score)
    if not passed and not reasons:
        reasons.append("score_below_threshold")

    return ScoreGateResult(passed=passed, score=score, reasons=reasons)
