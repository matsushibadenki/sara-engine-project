#!/usr/bin/env python3
"""Run a raw-text retention pilot on source-backed UD sentences."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

from sara_engine.language.semantic_echo import SparseSemanticEchoField
from sara_engine.language.semantic_events import LanguageEvent, SparseLanguageEventAdapter
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "role_labelled_heldout_cases_test_large.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_text_benchmark.json")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(case: Mapping[str, Any], *, gap_cap: int | None) -> Dict[str, Any]:
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    adapter = SparseLanguageEventAdapter(max_events=64)
    document_events = adapter.encode(str(case["document"]))
    head = str(edge["head"])
    dependent = str(edge["dependent"])
    query_events = adapter.encode(f"{head} {dependent}")
    gap = max(1, int(edge.get("distance", 1)))
    applied_gap = min(gap, gap_cap) if gap_cap is not None else gap
    field = SparseSemanticEchoField(tiers=("fast", "medium", "slow"), max_echoes=24, max_comparisons=32, enable_role_binding=False)
    events = tuple((1, event) for event in document_events) + tuple(
        (applied_gap if index == 0 else 1, event) for index, event in enumerate(query_events)
    )
    traces = field.run(events)
    decisions = [decision for trace in traces for decision in trace.decisions]
    expected = {head.casefold(), dependent.casefold()}
    hits = {str(decision.feature).casefold() for decision in decisions} & expected
    return {
        "case_id": str(case["case_id"]),
        "language": str(case["language"]),
        "task_family": str(case.get("task_family", "")),
        "distance": gap,
        "hit_count": len(hits),
        "correct": len(hits) >= 1,
        "active_echoes": max(trace.active_echoes for trace in traces),
        "comparisons": max(trace.comparisons for trace in traces),
        "state_bytes": field.serialized_state_bytes(),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_family: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_language: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["task_family"])].append(row)
        by_language[str(row["language"])].append(row)
    accuracy = lambda items: sum(bool(item["correct"]) for item in items) / max(1, len(items))
    return {
        "case_count": len(rows),
        "accuracy": accuracy(rows),
        "by_family": {key: accuracy(value) for key, value in sorted(by_family.items())},
        "by_language": {key: accuracy(value) for key, value in sorted(by_language.items())},
        "max_active_echoes": max((row["active_echoes"] for row in rows), default=0),
        "max_comparisons": max((row["comparisons"] for row in rows), default=0),
        "max_state_bytes": max((row["state_bytes"] for row in rows), default=0),
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    variants = {
        "baseline": None,
        "gap_cap_18": 18,
    }
    summaries = {name: _summary([_run(case, gap_cap=cap) for case in cases]) for name, cap in variants.items()}
    return {
        "schema": "sara-semantic-echo-ud-raw-text-benchmark-v1",
        "phase": "19/20",
        "observed_only": True,
        "external_assistance_disabled": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": summaries,
        "bounded_execution": all(
            summary["max_active_echoes"] <= 24 and summary["max_comparisons"] <= 32 and summary["max_state_bytes"] <= 4096
            for summary in summaries.values()
        ),
        "interpretation": [
            "Input is raw sentence text; dependency annotations are not encoded as event roles.",
            "The edge is used only to form a held-out retrieval target and to stratify errors.",
            "This is a retention pilot, not a full language-quality or production-promotion gate.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-path", default=DEFAULT_CASE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)
    report = build_report(case_path=args.case_path)
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"report_path": os.path.abspath(args.report_path), "bounded_execution": report["bounded_execution"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
