#!/usr/bin/env python3
"""Evaluate bounded exact tokenization without enabling a production path."""

from __future__ import annotations

import argparse
import importlib
import json
import os
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
    except ImportError:
        rust_encode = None
    rust_available = callable(rust_encode)
    ordered_merges = [
        pair
        for pair, _rank in sorted(
            tokenizer.merge_ranks.items(), key=lambda item: item[1]
        )
    ]
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

    cases: Dict[str, Any] = {}
    all_equivalent = True
    all_round_trip = True
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
        all_round_trip = (
            all_round_trip
            and base_round_trip
            and cached_round_trip
            and decode_equivalent
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
    checks = {
        "fixture_present": bool(rows),
        "token_ids_equivalent": all_equivalent,
        "decode_round_trip_preserved": all_round_trip,
        "repeated_outputs_equivalent": repeated_outputs == repeated_base,
        "spike_event_digest_equivalent": all_spike_equivalent,
        "malformed_utf8_rejected": malformed_utf8_rejected,
        "cache_entry_bound": cache["entries"] <= cache["max_entries"],
        "cache_byte_bound": cache["state_bytes"] <= cache["max_state_bytes"],
        "cache_reuse_observed": cache["hits"] > 0,
        "long_entry_bypass_observed": cache["bypassed"] > 0,
        "rust_scalar_reference_safe": (
            not rust_available or rust_equivalent
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
    return {
        "schema": "sara-phase27-tokenizer-acceleration-benchmark-v1",
        "passed": all(checks.values()),
        "observed_only": True,
        "production_path_changed": False,
        "rust_path_observed": rust_available and rust_equivalent,
        "rust_scalar_reference_available": rust_available,
        "rust_scalar_reference_equivalent": rust_equivalent,
        "gigatoken_path_observed": False,
        "tokenizer_fingerprint": adapter.fingerprint,
        "pretokenizer_identity": tokenizer.pretokenizer_identity(),
        "checks": checks,
        "cache": cache,
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
