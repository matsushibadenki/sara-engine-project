#!/usr/bin/env python3
"""Evaluate source-backed multilingual text through bounded Semantic Echo controls."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from sara_engine.language.semantic_echo import SparseSemanticEchoField
from sara_engine.language.semantic_events import LanguageEvent, SparseLanguageEventAdapter
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "heldout_cases.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_external_language_benchmark.json")
DEFAULT_TRACE_PATH = workspace_path("evaluation", "semantic_echo_external_language_traces.jsonl")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "semantic_echo_external_language_benchmark_summary.txt")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _events(case: Mapping[str, Any], *, max_events: int = 16) -> Tuple[Tuple[int, LanguageEvent], ...]:
    adapter = SparseLanguageEventAdapter(max_events=max_events)
    document_events = adapter.encode(str(case["document"]))
    # Keep both the opening definition and the closing operational detail within the
    # bounded event window; this is a deterministic text-window control, not a label lookup.
    document = document_events[:4] + document_events[-4:]
    query = adapter.encode(str(case["query"]))
    delayed_gap = 12 if str(case.get("task_type", "")) == "delayed" else 1
    return tuple((1, event) for event in document) + tuple(
        (delayed_gap if index == 0 else 1, event) for index, event in enumerate(query)
    )


def _keyword_match(decisions: Iterable[Any], keywords: Sequence[str]) -> bool:
    expected = {
        token.casefold()
        for keyword in keywords
        for token in re.findall(r"[\w]+|[^\w\s]", str(keyword), re.UNICODE)
    }
    return any(str(decision.feature).casefold() in expected for decision in decisions)


def _run(case: Mapping[str, Any], *, tiers: tuple[str, ...], role_binding: bool) -> Dict[str, Any]:
    field = SparseSemanticEchoField(
        tiers=tiers,
        max_echoes=9,
        max_comparisons=16,
        enable_role_binding=role_binding,
    )
    traces = field.run(_events(case))
    decisions = [decision for trace in traces for decision in trace.decisions]
    expected_behavior = str(case.get("expected_behavior", ""))
    keyword_match = _keyword_match(decisions, case.get("expected_keywords", []))
    correct = keyword_match if expected_behavior == "retrieve" else not keyword_match
    return {
        "case_id": str(case["case_id"]),
        "language": str(case.get("language", "")),
        "task_type": str(case.get("task_type", "")),
        "expected_behavior": expected_behavior,
        "correct": bool(correct),
        "keyword_match": bool(keyword_match),
        "abstained": bool(traces[-1].abstained),
        "active_echoes": max(trace.active_echoes for trace in traces),
        "comparisons": max(trace.comparisons for trace in traces),
        "updates": max(trace.updates for trace in traces),
        "state_bytes": field.serialized_state_bytes(),
        "decisions": [decision.__dict__ for decision in decisions],
    }


def _accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(bool(row["correct"]) for row in rows) / max(1, len(rows))


def _group_accuracy(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return {name: round(_accuracy(group), 6) for name, group in sorted(groups.items())}


def build_report(*, case_path: str = DEFAULT_CASE_PATH, trace_path: str = DEFAULT_TRACE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    single = [_run(case, tiers=("medium",), role_binding=False) for case in cases]
    multiscale = [_run(case, tiers=("fast", "medium", "slow"), role_binding=False) for case in cases]
    semantic_echo = [_run(case, tiers=("fast", "medium", "slow"), role_binding=True) for case in cases]
    replay = [_run(case, tiers=("fast", "medium", "slow"), role_binding=True) for case in cases]
    traces = [
        {"case": case, "single": one, "multiscale": multi, "semantic_echo": echo}
        for case, one, multi, echo in zip(cases, single, multiscale, semantic_echo)
    ]
    with open(ensure_parent_directory(trace_path), "w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    bounded = all(
        row["active_echoes"] <= 24
        and row["comparisons"] <= 32
        and row["state_bytes"] <= 4096
        for row in semantic_echo
    )
    replay_deterministic = all(
        left["correct"] == right["correct"]
        and left["decisions"] == right["decisions"]
        and left["state_bytes"] == right["state_bytes"]
        for left, right in zip(semantic_echo, replay)
    )
    abstention_integrity = all(
        row["correct"] for row in semantic_echo if row["expected_behavior"] == "abstain"
    )
    metrics = {
        "case_count": len(cases),
        "single_decay_accuracy": round(_accuracy(single), 6),
        "fixed_multiscale_accuracy": round(_accuracy(multiscale), 6),
        "semantic_echo_accuracy": round(_accuracy(semantic_echo), 6),
        "single_decay_by_language": _group_accuracy(single, "language"),
        "fixed_multiscale_by_language": _group_accuracy(multiscale, "language"),
        "semantic_echo_by_language": _group_accuracy(semantic_echo, "language"),
        "semantic_echo_by_task": _group_accuracy(semantic_echo, "task_type"),
        "abstention_integrity": float(abstention_integrity),
        "bounded_execution": float(bounded),
        "replay_determinism": float(replay_deterministic),
        "max_active_echoes": max((row["active_echoes"] for row in semantic_echo), default=0),
        "max_comparisons": max((row["comparisons"] for row in semantic_echo), default=0),
        "max_state_bytes": max((row["state_bytes"] for row in semantic_echo), default=0),
    }
    quality_gate = (
        metrics["semantic_echo_accuracy"] > metrics["single_decay_accuracy"]
        and metrics["semantic_echo_accuracy"] > metrics["fixed_multiscale_accuracy"]
    )
    metrics["quality_gate"] = float(quality_gate)
    return {
        "schema": "sara-semantic-echo-external-language-benchmark-v1",
        "phase": "19/20",
        "passed": bool(cases) and bool(bounded and replay_deterministic and abstention_integrity and quality_gate),
        "observed_only": True,
        "external_assistance_disabled": True,
        "source_isolation_preserved": all(
            str(case.get("evidence_scope", "")) == "independent_external"
            and str(case.get("derivation_stage", "")) == "post_source_split"
            for case in cases
        ),
        "case_path": os.path.abspath(case_path),
        "trace_path": os.path.abspath(trace_path),
        "metrics": metrics,
        "policy_notes": [
            "Cases are independent source-backed multilingual documents and are not repository fixtures.",
            "Text is converted to bounded surface events without an external parser or LLM.",
            "Single-decay and fixed multi-timescale SNN paths remain controls.",
            "A pass here does not establish physical energy advantage or production promotion.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-path", default=DEFAULT_CASE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args(argv)
    report = build_report(case_path=args.case_path, trace_path=args.trace_path)
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(ensure_parent_directory(args.summary_path), "w", encoding="utf-8") as handle:
        handle.write(f"External multilingual Semantic Echo benchmark: {'PASS' if report['passed'] else 'BLOCKED'}\n")
        for key, value in report["metrics"].items():
            handle.write(f"- {key}: {value}\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(args.report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
