#!/usr/bin/env python3
"""Run one frozen retrieval workload for paired physical energy measurement."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402


DEFAULT_CORPUS_PATH = processed_data_path("corpus.txt")
DEFAULT_OUTPUT_PATH = workspace_path("evaluation", "energy_pair_workload_result.json")


def _load_external_validity_module():
    path = os.path.join(PROJECT_ROOT, "scripts", "eval", "real_data_external_validity.py")
    spec = importlib.util.spec_from_file_location("energy_pair_external_validity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load real-data external-validity module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_retrieval_workload(
    *,
    system: str,
    corpus_path: str,
    max_docs: int,
    max_cases: int,
    repetitions: int,
    warmup_count: int,
) -> Dict[str, Any]:
    module = _load_external_validity_module()
    docs = module._load_corpus(corpus_path, limit=max_docs)
    tasks = module.build_real_data_tasks(docs, max_cases=max_cases)
    if not docs or not tasks:
        raise ValueError("Frozen retrieval workload requires non-empty docs and tasks.")
    retriever_type = (
        module.MetabolicSparseEventRetriever
        if system == "sara"
        else module.BM25OfflineProxyRetriever
    )

    for _ in range(max(0, int(warmup_count))):
        warmup_retriever = retriever_type(docs)
        module._score_retriever(warmup_retriever, tasks, docs)

    started = time.perf_counter()
    success_count = 0
    trial_count = 0
    event_cost = 0.0
    for _ in range(max(1, int(repetitions))):
        retriever = retriever_type(docs)
        score = module._score_retriever(retriever, tasks, docs)
        case_results = score.get("case_results", [])
        success_count += sum(
            1
            for row in case_results
            if isinstance(row, dict) and bool(row.get("correct", False))
        )
        trial_count += len(case_results)
        event_cost += float(score.get("avg_event_cost_proxy", 0.0) or 0.0)
    duration_seconds = time.perf_counter() - started
    success_rate = float(success_count) / float(max(1, trial_count))
    return {
        "schema": "sara-energy-pair-workload-result-v1",
        "task": "paired_retrieval",
        "system": system,
        "success_criterion_id": "retrieval-exact-document-index-v1",
        "success_count": success_count,
        "trial_count": trial_count,
        "success_rate": success_rate,
        "passed": success_rate >= 0.80,
        "duration_seconds": duration_seconds,
        "warmup_count": int(warmup_count),
        "measured_repetitions": int(repetitions),
        "doc_count": len(docs),
        "case_count": len(tasks),
        "avg_event_cost_proxy_across_repetitions": event_cost
        / float(max(1, int(repetitions))),
        "retriever": retriever_type.__name__,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen paired energy workload.")
    parser.add_argument("--system", choices=["sara", "ann"], required=True)
    parser.add_argument("--task", choices=["paired_retrieval"], default="paired_retrieval")
    parser.add_argument("--corpus-path", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--max-docs", type=int, default=256)
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--warmup-count", type=int, default=2)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_retrieval_workload(
        system=args.system,
        corpus_path=args.corpus_path,
        max_docs=args.max_docs,
        max_cases=args.max_cases,
        repetitions=args.repetitions,
        warmup_count=args.warmup_count,
    )
    report["corpus_path"] = os.path.abspath(args.corpus_path)
    output_path = ensure_parent_directory(args.output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
