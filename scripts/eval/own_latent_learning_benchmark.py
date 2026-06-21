#!/usr/bin/env python3
"""Observed-only benchmark for sparse own-latent learning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sara_engine.learning.own_latent import train_predictor_from_cases  # noqa: E402
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402
from scripts.eval.build_own_latent_rhm_fixture import build_cases, write_jsonl  # noqa: E402


DEFAULT_FIXTURE_PATH = processed_data_path("benchmark_fixtures", "own_latent_rhm_cases.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "own_latent_learning_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "own_latent_learning_benchmark_summary.txt")
DEFAULT_HISTORY_PATH = workspace_path("evaluation", "own_latent_learning_history.json")
DEFAULT_TRAIN_SIZES = (4, 8, 16, 32)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def ensure_fixture(path: str) -> List[Dict[str, Any]]:
    cases = read_jsonl(path)
    if cases:
        return cases
    cases = build_cases()
    write_jsonl(path, cases)
    return cases


def evaluate_train_size(cases: Sequence[Dict[str, Any]], train_size: int) -> Dict[str, Any]:
    predictor, baseline = train_predictor_from_cases(cases, train_size=train_size)
    eval_cases = [case for case in cases if str(case.get("split", "")) == "eval"]
    own_correct = 0
    token_correct = 0
    own_event_cost = 0
    token_event_cost = 0
    case_results: List[Dict[str, Any]] = []
    for case in eval_cases:
        expected = str(case.get("latent_group", ""))
        text = str(case.get("surface_text", ""))
        own_prediction = predictor.predict(text)
        token_label, token_score, token_cost = baseline.predict(text)
        own_ok = own_prediction.label == expected
        token_ok = token_label == expected
        own_correct += 1 if own_ok else 0
        token_correct += 1 if token_ok else 0
        own_event_cost += int(own_prediction.event_cost)
        token_event_cost += int(token_cost)
        case_results.append(
            {
                "case_id": case.get("case_id", ""),
                "expected": expected,
                "own_latent_label": own_prediction.label,
                "own_latent_score": own_prediction.score,
                "token_label": token_label,
                "token_score": token_score,
                "own_latent_correct": own_ok,
                "token_correct": token_ok,
                "own_latent_event_cost": own_prediction.event_cost,
                "token_event_cost": token_cost,
            }
        )

    count = max(1, len(eval_cases))
    own_accuracy = float(own_correct) / float(count)
    token_accuracy = float(token_correct) / float(count)
    avg_own_cost = float(own_event_cost) / float(count)
    avg_token_cost = float(token_event_cost) / float(count)
    return {
        "train_size": int(train_size),
        "eval_count": len(eval_cases),
        "own_latent_accuracy": round(own_accuracy, 6),
        "token_overlap_accuracy": round(token_accuracy, 6),
        "accuracy_delta": round(own_accuracy - token_accuracy, 6),
        "avg_own_latent_event_cost": round(avg_own_cost, 6),
        "avg_token_overlap_event_cost": round(avg_token_cost, 6),
        "state_budget_units": predictor.state_budget_units(),
        "sample_efficiency_ok": own_accuracy >= token_accuracy,
        "event_cost_bounded": predictor.state_budget_units() <= 4096 and avg_own_cost <= max(256.0, avg_token_cost * 2.0),
        "case_results": case_results,
    }


def build_report(cases: Sequence[Dict[str, Any]], train_sizes: Sequence[int]) -> Dict[str, Any]:
    evaluations = [evaluate_train_size(cases, train_size=size) for size in train_sizes]
    min_accuracy_delta = min(float(item["accuracy_delta"]) for item in evaluations) if evaluations else 0.0
    all_sample_efficiency_ok = all(bool(item["sample_efficiency_ok"]) for item in evaluations)
    all_event_cost_bounded = all(bool(item["event_cost_bounded"]) for item in evaluations)
    return {
        "schema": "sara-own-latent-learning-benchmark-v1",
        "suite_name": "OwnLatentLearningBenchmark",
        "observed_only": True,
        "case_count": len(cases),
        "train_sizes": [int(size) for size in train_sizes],
        "evaluations": evaluations,
        "metrics": {
            "own_latent_min_accuracy_delta": round(min_accuracy_delta, 6),
            "own_latent_sample_efficiency_ok": 1.0 if all_sample_efficiency_ok else 0.0,
            "own_latent_event_cost_bounded": 1.0 if all_event_cost_bounded else 0.0,
            "own_latent_max_state_budget_units": max(
                [int(item["state_budget_units"]) for item in evaluations] or [0]
            ),
        },
        "passed": bool(evaluations and all_sample_efficiency_ok and all_event_cost_bounded),
        "policy_notes": [
            "This is observed-only evidence and is not release-critical.",
            "The own-latent predictor uses sparse signatures and local co-occurrence updates.",
            "The token baseline is a comparison reference outside the production runtime path.",
            "No GPU, dense embedding matrix, or runtime backpropagation is required.",
        ],
    }


def write_report(report: Dict[str, Any], report_path: str, summary_path: str) -> Dict[str, str]:
    resolved_report = ensure_parent_directory(report_path)
    with open(resolved_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    resolved_summary = ensure_parent_directory(summary_path)
    lines = [
        f"Own-latent learning benchmark: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Cases: {report.get('case_count')}",
        f"Train sizes: {', '.join(str(item) for item in report.get('train_sizes', []))}",
        f"Min accuracy delta: {report.get('metrics', {}).get('own_latent_min_accuracy_delta')}",
        f"Max state budget units: {report.get('metrics', {}).get('own_latent_max_state_budget_units')}",
    ]
    with open(resolved_summary, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return {"report_path": resolved_report, "summary_path": resolved_summary}


def update_history(report: Dict[str, Any], history_path: str) -> None:
    resolved = ensure_parent_directory(history_path)
    history: List[Dict[str, Any]] = []
    if os.path.exists(resolved):
        try:
            with open(resolved, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                history = [item for item in payload if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            history = []
    history.append(
        {
            "passed": bool(report.get("passed")),
            "case_count": int(report.get("case_count", 0) or 0),
            "metrics": report.get("metrics", {}),
        }
    )
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(history[-20:], handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_train_sizes(raw: str) -> List[int]:
    sizes = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            sizes.append(int(item))
    return sizes or list(DEFAULT_TRAIN_SIZES)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sparse own-latent learning benchmark.")
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--history-path", default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--train-sizes", default=",".join(str(item) for item in DEFAULT_TRAIN_SIZES))
    parser.add_argument("--no-history-update", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cases = ensure_fixture(args.fixture_path)
    report = build_report(cases, parse_train_sizes(args.train_sizes))
    paths = write_report(report, args.report_path, args.summary_path)
    if not args.no_history_update:
        update_history(report, args.history_path)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "observed_only": report["observed_only"],
                "report_path": paths["report_path"],
                "summary_path": paths["summary_path"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
