#!/usr/bin/env python3
"""Diagnose raw-text tokenization versus observed UD token boundaries."""

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
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_tokenization_ablation.json")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(case: Mapping[str, Any], *, mode: str, max_echoes: int) -> Dict[str, Any]:
    adapter = SparseLanguageEventAdapter(max_events=64)
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    if mode == "surface":
        document_events = adapter.encode(str(case["document"]))
    elif mode == "ud_edge_oracle":
        observed = []
        for index, item in enumerate(case["dependency_or_role_edges"]):
            observed.extend((str(item["head"]), str(item["dependent"])))
        document_events = tuple(LanguageEvent(index, "orthographic", token) for index, token in enumerate(dict.fromkeys(observed)))
    else:
        raise ValueError(f"unknown mode: {mode}")
    query_events = adapter.encode(f"{edge['head']} {edge['dependent']}")
    gap = min(max(1, int(edge.get("distance", 1))), 18)
    field = SparseSemanticEchoField(tiers=("fast", "medium", "slow"), max_echoes=max_echoes, max_comparisons=16, enable_role_binding=False)
    traces = field.run(tuple((1, event) for event in document_events) + tuple((gap if i == 0 else 1, event) for i, event in enumerate(query_events)))
    expected = {str(edge["head"]).casefold(), str(edge["dependent"]).casefold()}
    decisions = [decision for trace in traces for decision in trace.decisions]
    hits = {str(decision.feature).casefold() for decision in decisions} & expected
    return {
        "case_id": str(case["case_id"]),
        "language": str(case["language"]),
        "task_family": str(case.get("task_family", "")),
        "correct": bool(hits),
        "active_echoes": max(trace.active_echoes for trace in traces),
        "comparisons": max(trace.comparisons for trace in traces),
        "state_bytes": field.serialized_state_bytes(),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def accuracy(items: Sequence[Mapping[str, Any]]) -> float:
        return sum(bool(item["correct"]) for item in items) / max(1, len(items))
    by_language: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[str(row["language"])].append(row)
    return {
        "accuracy": accuracy(rows),
        "by_language": {key: accuracy(value) for key, value in sorted(by_language.items())},
        "max_state_bytes": max((row["state_bytes"] for row in rows), default=0),
        "bounded": max((row["active_echoes"] for row in rows), default=0) <= 24
        and max((row["comparisons"] for row in rows), default=0) <= 32
        and max((row["state_bytes"] for row in rows), default=0) <= 4096,
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    variants = {}
    for mode in ("surface", "ud_edge_oracle"):
        for max_echoes in (6, 9, 12):
            name = f"{mode}_echo{max_echoes}"
            variants[name] = _summary([_run(case, mode=mode, max_echoes=max_echoes) for case in cases])
    return {
        "schema": "sara-semantic-echo-ud-tokenization-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "interpretation": [
            "surface uses the production raw-text adapter.",
            "ud_edge_oracle uses observed UD token boundaries and is an upper-bound diagnostic only.",
            "The oracle variant must not be used as promotion or production evidence.",
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
