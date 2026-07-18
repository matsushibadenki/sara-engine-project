#!/usr/bin/env python3
"""Ablate long-horizon decay and surface-identity handling on UD role labels."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

from sara_engine.language.semantic_echo import SparseSemanticEchoField
from sara_engine.language.semantic_events import LanguageEvent
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "role_labelled_heldout_cases.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_role_labelled_ablation.json")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(case: Mapping[str, Any], *, gap_cap: int | None, identity_aware: bool) -> Dict[str, Any]:
    edges = [edge for edge in case["dependency_or_role_edges"] if edge.get("relation") in {"nsubj", "obj", "iobj", "acl", "advcl"}]
    edge = max(edges or case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    relation = str(edge["relation"])
    head = str(edge["head"])
    dependent = str(edge["dependent"])
    head_feature = f"{head}@{edge['head_id']}" if identity_aware else head
    dependent_feature = f"{dependent}@{edge['dependent_id']}" if identity_aware else dependent
    gap = max(1, int(edge.get("distance", 1)))
    applied_gap = min(gap, gap_cap) if gap_cap is not None else gap
    field = SparseSemanticEchoField(
        tiers=("fast", "medium", "slow"),
        max_echoes=9,
        max_comparisons=16,
        enable_role_binding=True,
    )
    traces = field.run(
        (
            (1, LanguageEvent(1, "dependency", head_feature, role=relation)),
            (applied_gap, LanguageEvent(2, "dependency", dependent_feature, role=relation)),
        )
    )
    decisions = [decision for trace in traces for decision in trace.decisions]
    expected = f"{head_feature}->{dependent_feature}"
    return {
        "case_id": str(case["case_id"]),
        "language": str(case["language"]),
        "task_family": str(case.get("task_family", "")),
        "relation": relation,
        "original_gap": gap,
        "applied_gap": applied_gap,
        "expected": expected,
        "hit": any(decision.kind == "role_binding" and decision.feature == expected for decision in decisions),
        "active_echoes": max(trace.active_echoes for trace in traces),
        "comparisons": max(trace.comparisons for trace in traces),
        "state_bytes": field.serialized_state_bytes(),
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "recall": sum(bool(row["hit"]) for row in rows) / max(1, len(rows)),
        "by_language": {
            language: sum(row["language"] == language and row["hit"] for row in rows) / max(1, sum(row["language"] == language for row in rows))
            for language in sorted({str(row["language"]) for row in rows})
        },
        "by_task_family": {
            family: sum(row["task_family"] == family and row["hit"] for row in rows) / max(1, sum(row["task_family"] == family for row in rows))
            for family in sorted({str(row.get("task_family", "")) for row in rows})
        },
        "max_active_echoes": max((row["active_echoes"] for row in rows), default=0),
        "max_comparisons": max((row["comparisons"] for row in rows), default=0),
        "max_state_bytes": max((row["state_bytes"] for row in rows), default=0),
        "failure_cases": [row["case_id"] for row in rows if not row["hit"]],
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    variants = {
        "baseline": {"gap_cap": None, "identity_aware": False},
        "gap_cap_18": {"gap_cap": 18, "identity_aware": False},
        "identity_aware": {"gap_cap": None, "identity_aware": True},
        "gap_cap_18_identity_aware": {"gap_cap": 18, "identity_aware": True},
    }
    details: Dict[str, Any] = {}
    for name, options in variants.items():
        rows = [_run(case, **options) for case in cases]
        details[name] = {"summary": _summarize(rows), "rows": rows}
    baseline = details["baseline"]["summary"]["recall"]
    best = max(details[name]["summary"]["recall"] for name in variants)
    bounded = all(
        details[name]["summary"]["max_active_echoes"] <= 9
        and details[name]["summary"]["max_comparisons"] <= 16
        and details[name]["summary"]["max_state_bytes"] <= 4096
        for name in variants
    )
    return {
        "schema": "sara-semantic-echo-role-labelled-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_count": len(cases),
        "baseline_recall": baseline,
        "best_recall": best,
        "bounded_execution": bounded,
        "variants": details,
        "interpretation": [
            "gap_cap_18 is an ablation for a bounded long-horizon policy; it is not a production change.",
            "identity_aware uses observed token IDs to separate identical surface forms; it is not a semantic inference.",
            "No variant is promotion evidence without raw-text quality, held-out controls, latency, and energy review.",
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
    print(json.dumps({"best_recall": report["best_recall"], "report_path": os.path.abspath(args.report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
