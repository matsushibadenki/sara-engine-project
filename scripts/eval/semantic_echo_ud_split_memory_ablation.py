#!/usr/bin/env python3
"""Ablate split short/long memory for bounded raw-text retention."""

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
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_split_memory_ablation.json")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _field_events(field: SparseSemanticEchoField, events: Sequence[LanguageEvent], gap: int) -> tuple:
    return field.run(tuple((gap if index == 0 else 1, event) for index, event in enumerate(events)))


def _run(case: Mapping[str, Any], *, mode: str) -> Dict[str, Any]:
    adapter = SparseLanguageEventAdapter(max_events=64)
    document = adapter.encode(str(case["document"]))
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    query = adapter.encode(f"{edge['head']} {edge['dependent']}")
    gap = min(max(1, int(edge.get("distance", 1))), 18)
    fields: List[tuple[str, SparseSemanticEchoField, Sequence[LanguageEvent]]] = []
    if mode == "single_echo6":
        fields.append(("single", SparseSemanticEchoField(tiers=("fast", "medium", "slow"), max_echoes=6, max_comparisons=16), document))
    elif mode == "split_echo6_6":
        fields.append(("short", SparseSemanticEchoField(tiers=("fast", "medium"), max_echoes=6, max_comparisons=16), document[-8:]))
        fields.append(("long", SparseSemanticEchoField(tiers=("slow",), max_echoes=6, max_comparisons=16), document[::4]))
    elif mode == "split_echo9_3":
        fields.append(("short", SparseSemanticEchoField(tiers=("fast", "medium"), max_echoes=9, max_comparisons=16), document[-8:]))
        fields.append(("long", SparseSemanticEchoField(tiers=("slow",), max_echoes=3, max_comparisons=16), document[::4]))
    else:
        raise ValueError(f"unknown mode: {mode}")
    decisions = []
    all_traces = []
    for _, field, document_events in fields:
        traces = _field_events(field, document_events, 1)
        traces += _field_events(field, query, gap)
        all_traces.extend(traces)
        decisions.extend(decision for trace in traces for decision in trace.decisions)
    expected = {str(edge["head"]).casefold(), str(edge["dependent"]).casefold()}
    hits = {str(decision.feature).casefold() for decision in decisions} & expected
    return {
        "case_id": str(case["case_id"]),
        "language": str(case["language"]),
        "task_family": str(case.get("task_family", "")),
        "correct": bool(hits),
        "state_bytes": sum(field.serialized_state_bytes() for _, field, _ in fields),
        "active_echoes": max((trace.active_echoes for trace in all_traces), default=0),
        "comparisons": max((trace.comparisons for trace in all_traces), default=0),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def accuracy(items: Sequence[Mapping[str, Any]]) -> float:
        return sum(bool(item["correct"]) for item in items) / max(1, len(items))
    by_language: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_family: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[str(row["language"])].append(row)
        by_family[str(row["task_family"])].append(row)
    return {
        "accuracy": accuracy(rows),
        "by_language": {key: accuracy(value) for key, value in sorted(by_language.items())},
        "by_family": {key: accuracy(value) for key, value in sorted(by_family.items())},
        "max_state_bytes": max((row["state_bytes"] for row in rows), default=0),
        "max_active_echoes": max((row["active_echoes"] for row in rows), default=0),
        "max_comparisons": max((row["comparisons"] for row in rows), default=0),
        "bounded": max((row["state_bytes"] for row in rows), default=0) <= 4096
        and max((row["active_echoes"] for row in rows), default=0) <= 24
        and max((row["comparisons"] for row in rows), default=0) <= 32,
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    variants = {name: _summary([_run(case, mode=name) for case in cases]) for name in ("single_echo6", "split_echo6_6", "split_echo9_3")}
    return {
        "schema": "sara-semantic-echo-ud-split-memory-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "interpretation": [
            "Short memory keeps a bounded recent window; long memory receives a deterministic sparse sample.",
            "This is an ablation and does not modify production defaults.",
            "No raw-text quality or energy promotion is claimed from this result.",
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
    print(json.dumps({"report_path": os.path.abspath(args.report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
