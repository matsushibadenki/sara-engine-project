#!/usr/bin/env python3
"""Evaluate bounded exact tokenization without enabling a production path."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import statistics
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.edge.canonical_sparse_ir import replay_digest  # noqa: E402
from sara_engine.tokenization.exact_acceleration import (  # noqa: E402
    BoundedExactTokenizerAdapter,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)
from sara_engine.utils.tokenizer import SaraTokenizer  # noqa: E402

DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures", "phase27_tokenizer_conformance_cases.jsonl"
)
DEFAULT_OUTPUT = workspace_path(
    "evaluation", "phase27_tokenizer_acceleration_benchmark.json"
)


def load_cases(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _spike_digest(token_ids: Sequence[int]) -> str:
    events = [
        {
            "event_id": f"text-{index}",
            "timestep": index,
            "channel": "text",
            "spike_id": int(token_id),
            "modality": "text",
            "tags": ["phase27:tokenizer"],
        }
        for index, token_id in enumerate(token_ids)
    ]
    return replay_digest(events)


def _timed_encode(
    encode: Any,
    texts: Sequence[str],
) -> tuple[List[List[int]], int]:
    started = time.perf_counter_ns()
    outputs = [list(encode(text)) for text in texts]
    return outputs, time.perf_counter_ns() - started


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _snapshot_state_bytes(tokenizer: SaraTokenizer) -> int:
    payload = {
        "merges": [
            list(pair)
            for pair, _rank in sorted(
                tokenizer.merge_ranks.items(), key=lambda item: item[1]
            )
        ],
        "pretokenizer_identity": tokenizer.pretokenizer_identity(),
        "unknown_token_id": tokenizer.vocab.get("<unk>", 1),
        "vocab": tokenizer.vocab,
    }
    return len(
        json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    )


def _median_elapsed_ns(call: Any, repetitions: int) -> tuple[Any, int, List[int]]:
    samples: List[int] = []
    output: Any = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        output = call()
        samples.append(time.perf_counter_ns() - started)
    return output, int(statistics.median(samples)), samples


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    texts = [str(row["text"]) for row in rows]
    tokenizer = SaraTokenizer(
        vocab_size=2048,
        model_path=workspace_path(
            "evaluation", "phase27_unused_tokenizer_model.json"
        ),
        load_existing=False,
    )
    tokenizer.train(texts, save=False)
    adapter = BoundedExactTokenizerAdapter(
        tokenizer,
        max_entries=8,
        max_state_bytes=384,
        max_tokens_per_entry=32,
    )

    base_outputs, base_ns = _timed_encode(tokenizer.encode, texts)
    first_outputs, first_ns = _timed_encode(adapter.encode, texts)
    repeated_texts = [
        str(row["text"])
        for row in rows
        for _ in range(max(1, int(row.get("repetitions", 1))))
    ]
    repeated_base, repeated_base_ns = _timed_encode(
        tokenizer.encode,
        repeated_texts,
    )
    repeated_outputs, repeated_cache_ns = _timed_encode(
        adapter.encode,
        repeated_texts,
    )
    try:
        rust_core = importlib.import_module("sara_engine.sara_rust_core")
        rust_encode = getattr(
            rust_core, "tokenize_sara_bpe_pretokens", None
        )
        rust_batch_encode = getattr(
            rust_core, "batch_tokenize_sara_bpe_pretokens", None
        )
        rust_snapshot_class = getattr(
            rust_core, "FrozenSaraBpeTokenizer", None
        )
        rust_build_profile_fn = getattr(rust_core, "sara_rust_build_profile", None)
    except ImportError:
        rust_encode = None
        rust_batch_encode = None
        rust_snapshot_class = None
        rust_build_profile_fn = None
    rust_available = callable(rust_encode)
    rust_batch_available = callable(rust_batch_encode)
    ordered_merges = [
        pair
        for pair, _rank in sorted(
            tokenizer.merge_ranks.items(), key=lambda item: item[1]
        )
    ]
    snapshot_started = time.perf_counter_ns()
    rust_snapshot = (
        rust_snapshot_class(
            tokenizer.vocab,
            ordered_merges,
            tokenizer.vocab.get("<unk>", 1),
        )
        if callable(rust_snapshot_class)
        else None
    )
    rust_snapshot_construction_ns = time.perf_counter_ns() - snapshot_started
    rust_snapshot_available = rust_snapshot is not None
    rust_build_profile = (
        str(rust_build_profile_fn())
        if callable(rust_build_profile_fn)
        else "unknown"
    )
    rust_outputs = (
        [
            list(
                rust_encode(
                    tokenizer.pre_tokenize(text),
                    tokenizer.vocab,
                    ordered_merges,
                    tokenizer.vocab.get("<unk>", 1),
                )
            )
            for text in texts
        ]
        if rust_available
        else None
    )
    rust_equivalent = (
        rust_outputs == base_outputs if rust_outputs is not None else False
    )
    rust_boundary_calls = 0
    rust_batch_boundary_calls = 0
    rust_snapshot_boundary_calls = 0

    def rust_trace_encode(trace: Sequence[str]) -> List[List[int]]:
        nonlocal rust_boundary_calls
        outputs: List[List[int]] = []
        if not rust_available:
            return outputs
        for text in trace:
            rust_boundary_calls += 1
            outputs.append(
                list(
                    rust_encode(
                        tokenizer.pre_tokenize(text),
                        tokenizer.vocab,
                        ordered_merges,
                        tokenizer.vocab.get("<unk>", 1),
                    )
                )
            )
        return outputs

    def rust_batch_trace_encode(trace: Sequence[str]) -> List[List[int]]:
        nonlocal rust_batch_boundary_calls
        if not rust_batch_available:
            return []
        pretoken_batch = [tokenizer.pre_tokenize(text) for text in trace]
        rust_batch_boundary_calls += 1
        return [
            list(token_ids)
            for token_ids in rust_batch_encode(
                pretoken_batch,
                tokenizer.vocab,
                ordered_merges,
                tokenizer.vocab.get("<unk>", 1),
                1024,
                65536,
                1048576,
            )
        ]

    def rust_snapshot_trace_encode(trace: Sequence[str]) -> List[List[int]]:
        nonlocal rust_snapshot_boundary_calls
        if not rust_snapshot_available:
            return []
        rust_snapshot_boundary_calls += 1
        return [
            list(token_ids)
            for token_ids in rust_snapshot.batch_tokenize(
                [tokenizer.pre_tokenize(text) for text in trace],
                1024,
                65536,
                1048576,
            )
        ]

    resource_trace = texts + repeated_texts
    rss_before = _peak_rss_bytes()
    python_cold_outputs, python_cold_ns = _timed_encode(tokenizer.encode, resource_trace)
    rss_after_python = _peak_rss_bytes()
    rust_started = time.perf_counter_ns()
    rust_cold_outputs = rust_trace_encode(resource_trace)
    rust_cold_ns = time.perf_counter_ns() - rust_started
    rss_after_rust_cold = _peak_rss_bytes()
    rust_started = time.perf_counter_ns()
    rust_warm_outputs = rust_trace_encode(resource_trace)
    rust_warm_ns = time.perf_counter_ns() - rust_started
    rss_after_rust_warm = _peak_rss_bytes()
    rust_started = time.perf_counter_ns()
    rust_batch_cold_outputs = rust_batch_trace_encode(resource_trace)
    rust_batch_cold_ns = time.perf_counter_ns() - rust_started
    rss_after_rust_batch_cold = _peak_rss_bytes()
    rust_started = time.perf_counter_ns()
    rust_batch_warm_outputs = rust_batch_trace_encode(resource_trace)
    rust_batch_warm_ns = time.perf_counter_ns() - rust_started
    rss_after_rust_batch_warm = _peak_rss_bytes()
    rust_started = time.perf_counter_ns()
    rust_snapshot_cold_outputs = rust_snapshot_trace_encode(resource_trace)
    rust_snapshot_cold_ns = time.perf_counter_ns() - rust_started
    rss_after_rust_snapshot_cold = _peak_rss_bytes()
    rust_started = time.perf_counter_ns()
    rust_snapshot_warm_outputs = rust_snapshot_trace_encode(resource_trace)
    rust_snapshot_warm_ns = time.perf_counter_ns() - rust_started
    rss_after_rust_snapshot_warm = _peak_rss_bytes()
    snapshot_state_bytes = _snapshot_state_bytes(tokenizer)
    resource_spike_equivalent = bool(rust_available) and [
        _spike_digest(ids) for ids in rust_cold_outputs
    ] == [_spike_digest(ids) for ids in python_cold_outputs]
    batch_spike_equivalent = bool(rust_batch_available) and [
        _spike_digest(ids) for ids in rust_batch_cold_outputs
    ] == [_spike_digest(ids) for ids in python_cold_outputs]
    snapshot_spike_equivalent = bool(rust_snapshot_available) and [
        _spike_digest(ids) for ids in rust_snapshot_cold_outputs
    ] == [_spike_digest(ids) for ids in python_cold_outputs]
    median_trace = resource_trace * 10

    def python_median_call() -> List[List[int]]:
        return [list(tokenizer.encode(text)) for text in median_trace]

    def scalar_median_call() -> List[List[int]]:
        if not rust_available:
            return []
        return [
            list(
                rust_encode(
                    tokenizer.pre_tokenize(text),
                    tokenizer.vocab,
                    ordered_merges,
                    tokenizer.vocab.get("<unk>", 1),
                )
            )
            for text in median_trace
        ]

    def batch_median_call() -> List[List[int]]:
        if not rust_batch_available:
            return []
        return [
            list(token_ids)
            for token_ids in rust_batch_encode(
                [tokenizer.pre_tokenize(text) for text in median_trace],
                tokenizer.vocab,
                ordered_merges,
                tokenizer.vocab.get("<unk>", 1),
                1024,
                65536,
                1048576,
            )
        ]

    def snapshot_median_call() -> List[List[int]]:
        if not rust_snapshot_available:
            return []
        return [
            list(token_ids)
            for token_ids in rust_snapshot.batch_tokenize(
                [tokenizer.pre_tokenize(text) for text in median_trace],
                1024,
                65536,
                1048576,
            )
        ]

    python_median_call()
    scalar_median_call()
    batch_median_call()
    snapshot_median_call()
    median_repetitions = 7
    python_median_outputs, python_median_ns, python_samples = _median_elapsed_ns(
        python_median_call, median_repetitions
    )
    scalar_median_outputs, scalar_median_ns, scalar_samples = _median_elapsed_ns(
        scalar_median_call, median_repetitions
    )
    batch_median_outputs, batch_median_ns, batch_samples = _median_elapsed_ns(
        batch_median_call, median_repetitions
    )
    snapshot_median_outputs, snapshot_median_ns, snapshot_samples = _median_elapsed_ns(
        snapshot_median_call, median_repetitions
    )
    batch_median_speedup = (
        python_median_ns / batch_median_ns
        if rust_batch_available and batch_median_ns > 0
        else None
    )
    batch_performance_promotion_ready = bool(
        rust_batch_available
        and batch_median_outputs == python_median_outputs
        and batch_median_speedup is not None
        and batch_median_speedup > 1.05
    )
    snapshot_median_speedup = (
        python_median_ns / snapshot_median_ns
        if rust_snapshot_available and snapshot_median_ns > 0
        else None
    )
    snapshot_performance_promotion_ready = bool(
        rust_snapshot_available
        and snapshot_median_outputs == python_median_outputs
        and snapshot_median_speedup is not None
        and snapshot_median_speedup > 1.05
        and rust_build_profile == "release"
    )

    cases: Dict[str, Any] = {}
    all_equivalent = True
    all_decode_equivalent = True
    all_source_round_trip = True
    all_spike_equivalent = True
    for case_index, (row, base_ids, cached_ids) in enumerate(
        zip(rows, base_outputs, first_outputs)
    ):
        case_id = str(row["case_id"])
        source_text = str(row["text"])
        equivalent = base_ids == cached_ids
        base_round_trip = tokenizer.decode(base_ids) == source_text
        cached_round_trip = adapter.decode(cached_ids) == source_text
        decode_equivalent = (
            tokenizer.decode(base_ids) == adapter.decode(cached_ids)
        )
        spike_equivalent = _spike_digest(base_ids) == _spike_digest(cached_ids)
        all_equivalent = all_equivalent and equivalent
        all_decode_equivalent = all_decode_equivalent and decode_equivalent
        all_source_round_trip = (
            all_source_round_trip and base_round_trip and cached_round_trip
        )
        all_spike_equivalent = all_spike_equivalent and spike_equivalent
        cases[case_id] = {
            "language": str(row.get("language", "")),
            "input_bytes": len(source_text.encode("utf-8")),
            "token_count": len(base_ids),
            "token_ids_equivalent": equivalent,
            "base_round_trip": base_round_trip,
            "cached_round_trip": cached_round_trip,
            "decode_equivalent": decode_equivalent,
            "spike_digest_equivalent": spike_equivalent,
            "rust_scalar_token_ids_equivalent": (
                rust_outputs[case_index] == base_ids
                if rust_outputs is not None
                else None
            ),
        }

    try:
        adapter.encode_utf8(b"\xff")
        malformed_utf8_rejected = False
    except UnicodeDecodeError:
        malformed_utf8_rejected = True

    cache = adapter.stats()
    bypass_probe = BoundedExactTokenizerAdapter(
        tokenizer,
        max_entries=2,
        max_state_bytes=64,
        max_tokens_per_entry=1,
    )
    bypass_probe.encode(str(rows[-1]["text"]))
    bypass_probe_stats = bypass_probe.stats()
    checks = {
        "fixture_present": bool(rows),
        "token_ids_equivalent": all_equivalent,
        "decode_round_trip_preserved": all_decode_equivalent,
        "repeated_outputs_equivalent": repeated_outputs == repeated_base,
        "spike_event_digest_equivalent": all_spike_equivalent,
        "malformed_utf8_rejected": malformed_utf8_rejected,
        "cache_entry_bound": cache["entries"] <= cache["max_entries"],
        "cache_byte_bound": cache["state_bytes"] <= cache["max_state_bytes"],
        "cache_reuse_observed": cache["hits"] > 0,
        "long_entry_bypass_observed": bypass_probe_stats["bypassed"] > 0,
        "rust_scalar_reference_safe": (
            not rust_available or rust_equivalent
        ),
        "equal_trace_rust_outputs_equivalent": (
            not rust_available or rust_cold_outputs == python_cold_outputs
        ),
        "rust_warm_replay_equivalent": (
            not rust_available or rust_warm_outputs == rust_cold_outputs
        ),
        "downstream_replay_equivalent": (
            not rust_available or resource_spike_equivalent
        ),
        "rust_boundary_calls_accounted": (
            not rust_available
            or rust_boundary_calls == 2 * len(resource_trace)
        ),
        "rust_batch_reference_available": rust_batch_available,
        "rust_batch_outputs_equivalent": rust_batch_available
        and rust_batch_cold_outputs == python_cold_outputs,
        "rust_batch_warm_replay_equivalent": rust_batch_available
        and rust_batch_warm_outputs == rust_batch_cold_outputs,
        "rust_batch_downstream_replay_equivalent": batch_spike_equivalent,
        "rust_batch_boundary_calls_accounted": rust_batch_available
        and rust_batch_boundary_calls == 2,
        "rust_batch_boundary_reduced": rust_batch_available
        and rust_batch_boundary_calls < rust_boundary_calls,
        "rust_snapshot_reference_available": rust_snapshot_available,
        "rust_release_profile_observed": rust_build_profile == "release",
        "rust_snapshot_outputs_equivalent": rust_snapshot_available
        and rust_snapshot_cold_outputs == python_cold_outputs,
        "rust_snapshot_warm_replay_equivalent": rust_snapshot_available
        and rust_snapshot_warm_outputs == rust_snapshot_cold_outputs,
        "rust_snapshot_downstream_replay_equivalent": snapshot_spike_equivalent,
        "rust_snapshot_boundary_calls_accounted": rust_snapshot_available
        and rust_snapshot_boundary_calls == 2,
        "large_trace_scalar_equivalent": rust_available
        and scalar_median_outputs == python_median_outputs,
        "large_trace_batch_equivalent": rust_batch_available
        and batch_median_outputs == python_median_outputs,
        "large_trace_snapshot_equivalent": rust_snapshot_available
        and snapshot_median_outputs == python_median_outputs,
        "repeated_median_samples_complete": len(python_samples)
        == len(scalar_samples)
        == len(batch_samples)
        == len(snapshot_samples)
        == median_repetitions,
        "snapshot_state_bounded": snapshot_state_bytes <= 1_048_576,
        "peak_rss_growth_bounded": (
            max(
                rss_after_python,
                rss_after_rust_cold,
                rss_after_rust_warm,
                rss_after_rust_batch_cold,
                rss_after_rust_batch_warm,
                rss_after_rust_snapshot_cold,
                rss_after_rust_snapshot_warm,
            )
            - rss_before
            <= 67_108_864
        ),
        "rust_production_path_not_claimed": True,
        "gigatoken_path_not_claimed": True,
    }
    input_bytes = sum(len(text.encode("utf-8")) for text in texts)
    repeated_bytes = sum(
        len(text.encode("utf-8")) for text in repeated_texts
    )
    first_pass_speedup = base_ns / first_ns if first_ns > 0 else None
    repeated_speedup = (
        repeated_base_ns / repeated_cache_ns
        if repeated_cache_ns > 0
        else None
    )
    negative_results = []
    if first_pass_speedup is not None and first_pass_speedup <= 1.0:
        negative_results.append(
            "Bounded cache did not improve first-pass latency."
        )
    if repeated_speedup is not None and repeated_speedup <= 1.0:
        negative_results.append(
            "Bounded cache did not improve repeated-input latency."
        )
    if rust_batch_available and not batch_performance_promotion_ready:
        negative_results.append(
            "Batched Rust boundary did not exceed the frozen 1.05x repeated-median promotion threshold."
        )
    if rust_snapshot_available and not snapshot_performance_promotion_ready:
        negative_results.append(
            "Frozen Rust snapshot did not exceed the frozen 1.05x repeated-median promotion threshold."
        )
    return {
        "schema": "sara-phase27-tokenizer-acceleration-benchmark-v2",
        "passed": all(checks.values()),
        "observed_only": True,
        "production_path_changed": False,
        "rust_path_observed": rust_available and rust_equivalent,
        "rust_scalar_reference_available": rust_available,
        "rust_scalar_reference_equivalent": rust_equivalent,
        "rust_batch_reference_available": rust_batch_available,
        "rust_batch_reference_equivalent": rust_batch_available
        and rust_batch_cold_outputs == python_cold_outputs,
        "rust_batch_performance_promotion_ready": batch_performance_promotion_ready,
        "rust_snapshot_reference_available": rust_snapshot_available,
        "rust_snapshot_reference_equivalent": rust_snapshot_available
        and rust_snapshot_cold_outputs == python_cold_outputs,
        "rust_snapshot_performance_promotion_ready": snapshot_performance_promotion_ready,
        "rust_build_profile": rust_build_profile,
        "gigatoken_path_observed": False,
        "tokenizer_fingerprint": adapter.fingerprint,
        "pretokenizer_identity": tokenizer.pretokenizer_identity(),
        "checks": checks,
        "cache": cache,
        "bypass_probe": bypass_probe_stats,
        "resource_measurement": {
            "measurement_boundary": "equal-trace-end-to-end-python-pretoken-plus-scalar-merge-v1",
            "resource_trace_count": len(resource_trace),
            "python_boundary_calls": 0,
            "rust_boundary_calls": rust_boundary_calls,
            "rust_batch_boundary_calls": rust_batch_boundary_calls,
            "rust_snapshot_boundary_calls": rust_snapshot_boundary_calls,
            "rust_calls_per_pass": len(resource_trace) if rust_available else 0,
            "snapshot_state_bytes": snapshot_state_bytes,
            "rss_before_bytes": rss_before,
            "rss_after_python_bytes": rss_after_python,
            "rss_after_rust_cold_bytes": rss_after_rust_cold,
            "rss_after_rust_warm_bytes": rss_after_rust_warm,
            "rss_after_rust_batch_cold_bytes": rss_after_rust_batch_cold,
            "rss_after_rust_batch_warm_bytes": rss_after_rust_batch_warm,
            "rss_after_rust_snapshot_cold_bytes": rss_after_rust_snapshot_cold,
            "rss_after_rust_snapshot_warm_bytes": rss_after_rust_snapshot_warm,
            "peak_rss_delta_bytes": max(
                rss_after_python,
                rss_after_rust_cold,
                rss_after_rust_warm,
                rss_after_rust_batch_cold,
                rss_after_rust_batch_warm,
                rss_after_rust_snapshot_cold,
                rss_after_rust_snapshot_warm,
            ) - rss_before,
            "python_cold_elapsed_ns": python_cold_ns,
            "rust_cold_elapsed_ns": rust_cold_ns if rust_available else None,
            "rust_warm_elapsed_ns": rust_warm_ns if rust_available else None,
            "rust_batch_cold_elapsed_ns": (
                rust_batch_cold_ns if rust_batch_available else None
            ),
            "rust_batch_warm_elapsed_ns": (
                rust_batch_warm_ns if rust_batch_available else None
            ),
            "rust_snapshot_construction_elapsed_ns": rust_snapshot_construction_ns,
            "rust_snapshot_cold_elapsed_ns": (
                rust_snapshot_cold_ns if rust_snapshot_available else None
            ),
            "rust_snapshot_warm_elapsed_ns": (
                rust_snapshot_warm_ns if rust_snapshot_available else None
            ),
            "rust_cold_speedup_vs_python": (
                python_cold_ns / rust_cold_ns
                if rust_available and rust_cold_ns > 0
                else None
            ),
            "rust_warm_speedup_vs_python": (
                python_cold_ns / rust_warm_ns
                if rust_available and rust_warm_ns > 0
                else None
            ),
            "rust_batch_cold_speedup_vs_python": (
                python_cold_ns / rust_batch_cold_ns
                if rust_batch_available and rust_batch_cold_ns > 0
                else None
            ),
            "rust_batch_warm_speedup_vs_python": (
                python_cold_ns / rust_batch_warm_ns
                if rust_batch_available and rust_batch_warm_ns > 0
                else None
            ),
            "rust_snapshot_cold_speedup_vs_python": (
                python_cold_ns / rust_snapshot_cold_ns
                if rust_snapshot_available and rust_snapshot_cold_ns > 0
                else None
            ),
            "rust_snapshot_warm_speedup_vs_python": (
                python_cold_ns / rust_snapshot_warm_ns
                if rust_snapshot_available and rust_snapshot_warm_ns > 0
                else None
            ),
            "downstream_spike_replay_equivalent": resource_spike_equivalent,
            "batch_downstream_spike_replay_equivalent": batch_spike_equivalent,
            "snapshot_downstream_spike_replay_equivalent": snapshot_spike_equivalent,
            "median_trace_count": len(median_trace),
            "median_repetitions": median_repetitions,
            "python_median_elapsed_ns": python_median_ns,
            "rust_scalar_median_elapsed_ns": (
                scalar_median_ns if rust_available else None
            ),
            "rust_batch_median_elapsed_ns": (
                batch_median_ns if rust_batch_available else None
            ),
            "rust_snapshot_median_elapsed_ns": (
                snapshot_median_ns if rust_snapshot_available else None
            ),
            "rust_scalar_median_speedup_vs_python": (
                python_median_ns / scalar_median_ns
                if rust_available and scalar_median_ns > 0
                else None
            ),
            "rust_batch_median_speedup_vs_python": batch_median_speedup,
            "rust_snapshot_median_speedup_vs_python": snapshot_median_speedup,
            "python_median_samples_ns": python_samples,
            "rust_scalar_median_samples_ns": scalar_samples,
            "rust_batch_median_samples_ns": batch_samples,
            "rust_snapshot_median_samples_ns": snapshot_samples,
        },
        "cases": cases,
        "metrics": {
            "case_count": len(rows),
            "input_bytes": input_bytes,
            "repeated_input_bytes": repeated_bytes,
            "base_elapsed_ns": base_ns,
            "cache_first_pass_elapsed_ns": first_ns,
            "repeated_base_elapsed_ns": repeated_base_ns,
            "cache_repeated_pass_elapsed_ns": repeated_cache_ns,
            "first_pass_speedup_vs_base": first_pass_speedup,
            "repeated_speedup_vs_base": repeated_speedup,
        },
        "negative_results": negative_results,
        "policy_notes": [
            "Timing is diagnostic and is not a promotion claim.",
            "Peak RSS is a process high-water observation and may report zero incremental growth on a warmed process.",
            "Boundary-call counts are explicit adapter invocations, not profiler-derived CPU instructions.",
            "The batched Rust path remains optional and production routing is unchanged.",
            (
                "Source-text round trip was observed."
                if all_source_round_trip
                else "Source-text round trip is not claimed where the frozen pretokenizer normalizes boundary whitespace; candidate decode still matches the reference decode exactly."
            ),
            "The tokenizer snapshot and bounded cache remain optional.",
            (
                "Rust scalar BPE equivalence was observed."
                if rust_available and rust_equivalent
                else "Rust scalar BPE equivalence remains unresolved."
            ),
            "Gigatoken equivalence remains unresolved.",
            "No physical-energy claim is made.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report(load_cases(args.fixture_path))
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
