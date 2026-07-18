#!/usr/bin/env python3
"""Ablate bounded text windows and echo occupancy on UD raw-text cases."""

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
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_text_bounded_ablation.json")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(case: Mapping[str, Any], *, window: int, max_echoes: int, gap_cap: int | None) -> Dict[str, Any]:
    adapter = SparseLanguageEventAdapter(max_events=window)
    document_events = adapter.encode(str(case["document"]))
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    query_events = adapter.encode(f"{edge['head']} {edge['dependent']}")
    gap = max(1, int(edge.get("distance", 1)))
    applied_gap = min(gap, gap_cap) if gap_cap is not None else gap
    field = SparseSemanticEchoField(tiers=("fast", "medium", "slow"), max_echoes=max_echoes, max_comparisons=16, enable_role_binding=False)
    traces = field.run(tuple((1, event) for event in document_events) + tuple((applied_gap if i == 0 else 1, event) for i, event in enumerate(query_events)))
    decisions = [decision for trace in traces for decision in trace.decisions]
    expected = {str(edge["head"]).casefold(), str(edge["dependent"]).casefold()}
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
    languages: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    families: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        languages[str(row["language"])].append(row)
        families[str(row["task_family"])].append(row)
    return {
        "accuracy": accuracy(rows),
        "by_language": {key: accuracy(value) for key, value in sorted(languages.items())},
        "by_family": {key: accuracy(value) for key, value in sorted(families.items())},
        "max_active_echoes": max((row["active_echoes"] for row in rows), default=0),
        "max_comparisons": max((row["comparisons"] for row in rows), default=0),
        "max_state_bytes": max((row["state_bytes"] for row in rows), default=0),
        "bounded": max((row["active_echoes"] for row in rows), default=0) <= 24
        and max((row["comparisons"] for row in rows), default=0) <= 32
        and max((row["state_bytes"] for row in rows), default=0) <= 4096,
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    configs = {
        "window8_echo6_gap18": {"window": 8, "max_echoes": 6, "gap_cap": 18},
        "window12_echo6_gap18": {"window": 12, "max_echoes": 6, "gap_cap": 18},
        "window16_echo9_gap18": {"window": 16, "max_echoes": 9, "gap_cap": 18},
        "window32_echo12_gap18": {"window": 32, "max_echoes": 12, "gap_cap": 18},
        "window48_echo18_gap18": {"window": 48, "max_echoes": 18, "gap_cap": 18},
        "window64_echo24_gap18": {"window": 64, "max_echoes": 24, "gap_cap": 18},
    }
    variants = {
        name: _summary([_run(case, **options) for case in cases])
        for name, options in configs.items()
    }
    bounded = {name: value["bounded"] for name, value in variants.items()}
    best_bounded = max((value["accuracy"] for value in variants.values() if value["bounded"]), default=0.0)
    return {
        "schema": "sara-semantic-echo-ud-text-bounded-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "bounded_variants": bounded,
        "best_bounded_accuracy": best_bounded,
        "interpretation": [
            "All variants use raw sentence text and no external parser or LLM.",
            "A smaller window or echo budget is an evaluation ablation, not a production default change.",
            "Quality gains are not promotion evidence without held-out controls, latency, and energy review.",
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
    print(json.dumps({"best_bounded_accuracy": report["best_bounded_accuracy"], "report_path": os.path.abspath(args.report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
