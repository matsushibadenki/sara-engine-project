#!/usr/bin/env python3
"""Classify raw-text retrieval failures without changing evaluation behavior."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence

from sara_engine.language.semantic_echo import SparseSemanticEchoField
from sara_engine.language.semantic_events import LanguageEvent, SparseLanguageEventAdapter
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "role_labelled_heldout_cases_test_large.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_raw_failure_modes.json")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(case: Mapping[str, Any], *, gap_cap: int | None) -> Dict[str, Any]:
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    adapter = SparseLanguageEventAdapter(max_events=64)
    document_events = adapter.encode(str(case["document"]))
    head = str(edge["head"]).casefold()
    dependent = str(edge["dependent"]).casefold()
    query_events = adapter.encode(f"{head} {dependent}")
    gap = max(1, int(edge.get("distance", 1)))
    applied_gap = min(gap, gap_cap) if gap_cap is not None else gap
    field = SparseSemanticEchoField(tiers=("fast", "medium", "slow"), max_echoes=24, max_comparisons=32, enable_role_binding=False)
    traces = field.run(tuple((1, event) for event in document_events) + tuple((applied_gap if index == 0 else 1, event) for index, event in enumerate(query_events)))
    decisions = {str(decision.feature).casefold() for trace in traces for decision in trace.decisions}
    return {
        "case_id": str(case["case_id"]),
        "language": str(case["language"]),
        "distance": gap,
        "head": head,
        "dependent": dependent,
        "head_in_document": any(event.feature == head for event in document_events),
        "dependent_in_document": any(event.feature == dependent for event in document_events),
        "head_decision": head in decisions,
        "dependent_decision": dependent in decisions,
        "correct": bool({head, dependent} & decisions),
        "gap_capped": gap_cap is not None and gap > gap_cap,
        "same_surface": head == dependent,
    }


def _categories(row: Mapping[str, Any]) -> List[str]:
    categories = []
    if not row["head_in_document"] or not row["dependent_in_document"]:
        categories.append("endpoint_not_observed_in_document")
    if row["same_surface"]:
        categories.append("identical_surface_endpoints")
    if int(row["distance"]) > 18:
        categories.append("declared_long_distance_gt18")
    if row["head_in_document"] and row["dependent_in_document"] and not row["correct"]:
        categories.append("both_endpoints_present_but_not_recalled")
    if not categories:
        categories.append("other")
    return categories


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_language: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    category_counts = Counter()
    for row in rows:
        by_language[str(row["language"])].append(row)
    failures = [row for row in rows if not row["correct"]]
    for row in failures:
        category_counts.update(_categories(row))
    return {
        "case_count": len(rows),
        "recall": sum(bool(row["correct"]) for row in rows) / max(1, len(rows)),
        "failure_count": len(failures),
        "category_counts": dict(sorted(category_counts.items())),
        "failure_by_language": {
            language: sum(not row["correct"] for row in items)
            for language, items in sorted(by_language.items())
        },
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    variants = {}
    for name, cap in (("baseline", None), ("gap_cap_18", 18)):
        rows = [_run(case, gap_cap=cap) for case in cases]
        variants[name] = _summarize(rows)
    baseline_rows = [_run(case, gap_cap=None) for case in cases]
    return {
        "schema": "sara-semantic-echo-ud-raw-failure-modes-v1",
        "phase": "19/20",
        "observed_only": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "baseline_failure_cases": [
            {
                "case_id": row["case_id"],
                "language": row["language"],
                "distance": row["distance"],
                "categories": _categories(row),
            }
            for row in baseline_rows
            if not row["correct"]
        ],
        "interpretation": [
            "Categories are descriptive diagnostics from observed text and edge metadata.",
            "Endpoint presence does not imply correct dependency role binding.",
            "No production behavior is changed by this analysis.",
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
