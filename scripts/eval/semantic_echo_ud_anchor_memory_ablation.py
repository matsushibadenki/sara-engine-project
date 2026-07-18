#!/usr/bin/env python3
"""Measure bounded surface-anchor memory on independently labelled UD text."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

from sara_engine.language.semantic_events import SparseLanguageEventAdapter
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "role_labelled_heldout_cases_test_large.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_anchor_memory_ablation.json")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(case: Mapping[str, Any], *, capacity: int) -> Dict[str, Any]:
    adapter = SparseLanguageEventAdapter(max_events=256)
    document = adapter.encode(str(case["document"]))
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    expected = {str(edge["head"]).casefold(), str(edge["dependent"]).casefold()}

    # Diagnostic cache only: no role labels, dependency labels, or gold edge
    # positions are used to select entries.
    anchors: Dict[str, int] = {}
    time = 0
    for event in document:
        if event.axis != "orthographic":
            continue
        time += 1
        anchors[event.feature] = time
        if len(anchors) > capacity:
            oldest = min(anchors.items(), key=lambda item: (item[1], item[0]))[0]
            del anchors[oldest]

    hits = expected & set(anchors)
    state = {"schema": "sara-anchor-memory-v1", "anchors": sorted(anchors.items())}
    state_bytes = len(json.dumps(state, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "case_id": str(case["case_id"]),
        "language": str(case["language"]),
        "task_family": str(case.get("task_family", "")),
        "correct": bool(hits),
        "matched_count": len(hits),
        "capacity": capacity,
        "state_bytes": state_bytes,
        "active_anchors": len(anchors),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def accuracy(items: Sequence[Mapping[str, Any]]) -> float:
        return sum(bool(item["correct"]) for item in items) / max(1, len(items))

    by_language: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_family: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[str(row["language"])].append(row)
        by_family[str(row["task_family"])].append(row)
    max_state = max((int(row["state_bytes"]) for row in rows), default=0)
    return {
        "accuracy": accuracy(rows),
        "by_language": {key: accuracy(value) for key, value in sorted(by_language.items())},
        "by_family": {key: accuracy(value) for key, value in sorted(by_family.items())},
        "max_state_bytes": max_state,
        "max_active_anchors": max((int(row["active_anchors"]) for row in rows), default=0),
        "bounded": max_state <= 4096,
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    capacities = (8, 16, 32, 64)
    variants = {f"anchor_cache_{capacity}": _summary([_run(case, capacity=capacity) for case in cases]) for capacity in capacities}
    return {
        "schema": "sara-semantic-echo-ud-anchor-memory-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "diagnostic_upper_bound": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "interpretation": [
            "The cache stores only surface orthographic features observed in the document.",
            "The query is used only for evaluation matching; no dependency or role label selects memory entries.",
            "This lexical anchor cache is a diagnostic upper bound and is not production evidence for role binding.",
            "No production defaults or model behavior are changed by this ablation.",
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
