#!/usr/bin/env python3
"""Sanity-check Semantic Echo role binding on independently annotated UD edges."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from sara_engine.language.semantic_echo import SparseSemanticEchoField
from sara_engine.language.semantic_events import LanguageEvent
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "role_labelled_heldout_cases.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_role_labelled_benchmark.json")
DEFAULT_TRACE_PATH = workspace_path("evaluation", "semantic_echo_role_labelled_traces.jsonl")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(case: Mapping[str, Any], *, role_binding: bool) -> Dict[str, Any]:
    edges = [edge for edge in case["dependency_or_role_edges"] if edge.get("relation") in {"nsubj", "obj", "iobj", "acl", "advcl"}]
    edge = max(edges or case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    relation = str(edge["relation"])
    head = str(edge["head"])
    dependent = str(edge["dependent"])
    field = SparseSemanticEchoField(
        tiers=("fast", "medium", "slow"),
        max_echoes=9,
        max_comparisons=16,
        enable_role_binding=role_binding,
    )
    traces = field.run(
        (
            (1, LanguageEvent(1, "dependency", head, role=relation)),
            (max(1, int(edge.get("distance", 1))), LanguageEvent(2, "dependency", dependent, role=relation)),
        )
    )
    decisions = [decision for trace in traces for decision in trace.decisions]
    expected = f"{head}->{dependent}"
    hit = any(decision.kind == "role_binding" and decision.feature == expected for decision in decisions)
    if hit:
        error_reason = "none"
    elif head == dependent:
        error_reason = "identical_head_dependent_surface"
    elif int(edge.get("distance", 0)) > 18:
        error_reason = "echo_expired_at_declared_gap"
    else:
        error_reason = "binding_not_emitted"
    return {
        "case_id": str(case["case_id"]),
        "language": str(case["language"]),
        "relation": relation,
        "distance": int(edge.get("distance", 0)),
        "expected_binding": expected,
        "role_binding": role_binding,
        "hit": bool(hit),
        "error_reason": error_reason,
        "active_echoes": max(trace.active_echoes for trace in traces),
        "comparisons": max(trace.comparisons for trace in traces),
        "state_bytes": field.serialized_state_bytes(),
        "decisions": [decision.__dict__ for decision in decisions],
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH, trace_path: str = DEFAULT_TRACE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    control = [_run(case, role_binding=False) for case in cases]
    role = [_run(case, role_binding=True) for case in cases]
    replay = [_run(case, role_binding=True) for case in cases]
    with open(ensure_parent_directory(trace_path), "w", encoding="utf-8") as handle:
        for left, right in zip(control, role):
            handle.write(json.dumps({"control": left, "role_binding": right}, ensure_ascii=False, sort_keys=True) + "\n")
    role_hits = sum(row["hit"] for row in role)
    control_hits = sum(row["hit"] for row in control)
    bounded = all(row["active_echoes"] <= 9 and row["comparisons"] <= 16 and row["state_bytes"] <= 4096 for row in role)
    metrics = {
        "case_count": len(cases),
        "role_binding_recall": role_hits / max(1, len(role)),
        "control_role_binding_recall": control_hits / max(1, len(control)),
        "role_binding_improves_control": float(role_hits > control_hits),
        "replay_determinism": float(all(left == right for left, right in zip(role, replay))),
        "bounded_execution": float(bounded),
        "by_language": {
            language: {
                "case_count": sum(row["language"] == language for row in role),
                "role_binding_hits": sum(row["language"] == language and row["hit"] for row in role),
            }
            for language in sorted({str(row["language"]) for row in role})
        },
        "relation_counts": dict(Counter(row["relation"] for row in role)),
        "error_reasons": dict(Counter(row["error_reason"] for row in role if not row["hit"])),
        "error_cases": [
            {
                "case_id": row["case_id"],
                "language": row["language"],
                "relation": row["relation"],
                "distance": row["distance"],
                "expected_binding": row["expected_binding"],
                "error_reason": row["error_reason"],
            }
            for row in role
            if not row["hit"]
        ],
    }
    passed = bool(cases) and bool(metrics["role_binding_improves_control"] and bounded and metrics["replay_determinism"])
    return {
        "schema": "sara-semantic-echo-role-labelled-benchmark-v1",
        "phase": "19/20",
        "passed": passed,
        "observed_only": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "trace_path": os.path.abspath(trace_path),
        "metrics": metrics,
        "policy_notes": [
            "Dependency edges are observed annotations from UD test splits.",
            "This benchmark checks structural role binding only; it does not claim raw-text language quality.",
            "No external parser or LLM is used at evaluation time.",
            "A pass does not establish energy advantage or production promotion.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-path", default=DEFAULT_CASE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    args = parser.parse_args(argv)
    report = build_report(case_path=args.case_path, trace_path=args.trace_path)
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(args.report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
