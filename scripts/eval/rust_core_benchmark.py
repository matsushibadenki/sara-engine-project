#!/usr/bin/env python3
"""Compare Rust sparse-runtime exports with Python reference paths when available."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_REPORT_PATH = workspace_path("evaluation", "rust_core_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "rust_core_benchmark_summary.txt")


def python_sdr_overlap(sdr_a: Sequence[int], sdr_b: Sequence[int]) -> float:
    set_a = set(sdr_a)
    set_b = set(sdr_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / float(max(len(set_a), len(set_b)))


def python_sparse_propagate_threshold(
    active_spikes: Sequence[int],
    weights: Sequence[Mapping[int, float] | Sequence[Tuple[int, float]] | Sequence[float]],
    out_size: int,
    threshold: float,
) -> List[int]:
    potentials = [0.0] * out_size
    for spike in active_spikes:
        if spike >= len(weights):
            continue
        targets = weights[spike]
        if isinstance(targets, Mapping):
            items = targets.items()
        elif targets and isinstance(targets[0], tuple):  # type: ignore[index]
            items = targets  # type: ignore[assignment]
        else:
            items = enumerate(targets)  # type: ignore[arg-type]
        for target_id, weight in items:
            if int(target_id) < out_size:
                potentials[int(target_id)] += float(weight)
    return [idx for idx, potential in enumerate(potentials) if potential >= threshold]


def python_build_direct_synapses(
    tokens: Sequence[int], context_window: int
) -> Dict[int, Dict[int, Dict[int, float]]]:
    co_occurrence: DefaultDict[int, DefaultDict[int, DefaultDict[int, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    unigram_counts: DefaultDict[int, int] = defaultdict(int)
    total_tokens = len(tokens)
    for index, token in enumerate(tokens):
        unigram_counts[token] += 1
        end_index = min(index + context_window + 1, total_tokens)
        for next_index in range(index + 1, end_index):
            delay = next_index - index
            co_occurrence[delay][token][tokens[next_index]] += 1.0

    synapses: Dict[int, Dict[int, Dict[int, float]]] = {}
    for delay, pre_dict in co_occurrence.items():
        synapses[delay] = {}
        for pre_token, posts in pre_dict.items():
            pre_count = float(unigram_counts[pre_token])
            synapses[delay][pre_token] = {}
            for post_token, count in posts.items():
                post_count = float(unigram_counts[post_token])
                synapses[delay][pre_token][post_token] = count / math.sqrt(pre_count * post_count)
    return synapses


def python_batch_tokens_to_sdr(
    batch_tokens: Sequence[Sequence[int]], vocab_size: int, sdr_density: float, seed: int
) -> List[List[List[int]]]:
    sdr_size = max(1, math.ceil(vocab_size * sdr_density))
    batch_sdrs: List[List[List[int]]] = []
    for sequence in batch_tokens:
        sequence_sdrs: List[List[int]] = []
        for token in sequence:
            state = (int(seed) ^ int(token)) & ((1 << 64) - 1)
            values = []
            for _ in range(sdr_size):
                state = (
                    state * 6364136223846793005 + 1442695040888963407
                ) & ((1 << 64) - 1)
                values.append((state >> 32) % vocab_size)
            sdr = sorted(set(values))
            sequence_sdrs.append(sdr)
        batch_sdrs.append(sequence_sdrs)
    return batch_sdrs


def normalize_nested_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            int(key): normalize_nested_mapping(inner)
            for key, inner in sorted(value.items(), key=lambda item: int(item[0]))
        }
    if isinstance(value, list):
        return [normalize_nested_mapping(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_nested_mapping(item) for item in value)
    if isinstance(value, float):
        return round(value, 7)
    return value


def timed_call(fn: Callable[[], Any], iterations: int) -> Tuple[Any, float]:
    result: Any = None
    started = time.perf_counter()
    for _ in range(iterations):
        result = fn()
    elapsed = time.perf_counter() - started
    return result, elapsed


def load_rust_core() -> Any | None:
    try:
        return importlib.import_module("sara_engine.sara_rust_core")
    except Exception:
        try:
            return importlib.import_module("sara_rust_core")
        except Exception:
            return None


def benchmark_case(
    *,
    name: str,
    python_fn: Callable[[], Any],
    rust_fn: Callable[[], Any] | None,
    iterations: int,
) -> Dict[str, Any]:
    python_result, python_seconds = timed_call(python_fn, iterations)
    case: Dict[str, Any] = {
        "name": name,
        "iterations": iterations,
        "python_seconds": python_seconds,
        "python_result_fingerprint": repr(normalize_nested_mapping(python_result))[:500],
        "rust_available": rust_fn is not None,
    }
    if rust_fn is None:
        case.update(
            {
                "rust_seconds": None,
                "speedup_vs_python": None,
                "outputs_equivalent": None,
            }
        )
        return case
    rust_result, rust_seconds = timed_call(rust_fn, iterations)
    outputs_equivalent = normalize_nested_mapping(python_result) == normalize_nested_mapping(rust_result)
    case.update(
        {
            "rust_seconds": rust_seconds,
            "rust_result_fingerprint": repr(normalize_nested_mapping(rust_result))[:500],
            "speedup_vs_python": python_seconds / rust_seconds if rust_seconds > 0.0 else None,
            "outputs_equivalent": outputs_equivalent,
        }
    )
    return case


def build_cases(rust_core: Any | None, iterations: int) -> List[Dict[str, Any]]:
    active_spikes = [0, 3, 5, 7]
    sparse_weights = [
        {1: 0.4, 2: 0.7},
        {},
        {},
        {2: 0.2, 4: 0.9},
        {},
        {4: 0.3, 5: 0.8},
        {},
        {5: 0.25, 6: 1.1},
    ]
    tokens = [idx % 17 for idx in range(240)]
    batch_tokens = [[idx % 31 for idx in range(32)], [idx % 13 for idx in range(32)]]

    rust_overlap = None if rust_core is None else lambda: rust_core.calculate_sdr_overlap(
        [1, 2, 3, 5, 8, 13], [3, 5, 8, 21]
    )
    rust_propagate = None if rust_core is None else lambda: rust_core.sparse_propagate_threshold(
        active_spikes, sparse_weights, 8, 0.8
    )
    rust_direct = None if rust_core is None else lambda: rust_core.build_direct_synapses(tokens, 4)
    rust_batch = None if rust_core is None else lambda: rust_core.batch_tokens_to_sdr(
        batch_tokens, 128, 0.0625, 1234
    )

    return [
        benchmark_case(
            name="sdr_overlap",
            python_fn=lambda: python_sdr_overlap([1, 2, 3, 5, 8, 13], [3, 5, 8, 21]),
            rust_fn=rust_overlap,
            iterations=iterations,
        ),
        benchmark_case(
            name="sparse_propagate_threshold",
            python_fn=lambda: python_sparse_propagate_threshold(active_spikes, sparse_weights, 8, 0.8),
            rust_fn=rust_propagate,
            iterations=iterations,
        ),
        benchmark_case(
            name="build_direct_synapses",
            python_fn=lambda: python_build_direct_synapses(tokens, 4),
            rust_fn=rust_direct,
            iterations=max(1, iterations // 20),
        ),
        benchmark_case(
            name="batch_tokens_to_sdr",
            python_fn=lambda: python_batch_tokens_to_sdr(batch_tokens, 128, 0.0625, 1234),
            rust_fn=rust_batch,
            iterations=iterations,
        ),
    ]


def build_report(iterations: int = 50) -> Dict[str, Any]:
    rust_core = load_rust_core()
    cases = build_cases(rust_core, iterations)
    comparable_cases = [case for case in cases if case["rust_available"]]
    equivalent_cases = [
        case for case in comparable_cases if case.get("outputs_equivalent") is True
    ]
    speedups = [
        float(case["speedup_vs_python"])
        for case in comparable_cases
        if isinstance(case.get("speedup_vs_python"), (int, float))
    ]
    return {
        "schema": "sara-rust-core-benchmark-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rust_extension_available": rust_core is not None,
        "case_count": len(cases),
        "comparable_case_count": len(comparable_cases),
        "output_equivalence_passed": bool(comparable_cases) and len(equivalent_cases) == len(comparable_cases),
        "min_speedup_vs_python": min(speedups) if speedups else None,
        "cases": cases,
        "policy_notes": [
            "Benchmarks compare sparse event primitives only.",
            "Dense ANN baselines are not part of this runtime path.",
            "Reports are written only under workspace/evaluation.",
        ],
    }


def summarize_report(report: Mapping[str, Any]) -> str:
    lines = [
        "Rust core benchmark:",
        f"Rust extension available: {report.get('rust_extension_available')}",
        f"Comparable cases: {report.get('comparable_case_count')}/{report.get('case_count')}",
        f"Output equivalence passed: {report.get('output_equivalence_passed')}",
        f"Minimum speedup vs Python: {report.get('min_speedup_vs_python')}",
    ]
    for case in report.get("cases", []):
        lines.append(
            f"- {case.get('name')}: rust_available={case.get('rust_available')}, "
            f"equivalent={case.get('outputs_equivalent')}, speedup={case.get('speedup_vs_python')}"
        )
    return "\n".join(lines) + "\n"


def write_report(report: Mapping[str, Any], report_path: str, summary_path: str) -> Dict[str, str]:
    resolved_report = ensure_parent_directory(report_path)
    resolved_summary = ensure_parent_directory(summary_path)
    with open(resolved_report, "w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(resolved_summary, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))
    return {"report_path": resolved_report, "summary_path": resolved_summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(iterations=max(1, args.iterations))
    paths = write_report(report, args.report_path, args.summary_path)
    print(json.dumps({"rust_extension_available": report["rust_extension_available"], **paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
