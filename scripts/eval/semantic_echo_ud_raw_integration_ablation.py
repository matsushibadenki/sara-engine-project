#!/usr/bin/env python3
"""Safely ablate raw-text tokenization and bounded gap handling."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.language.semantic_echo import SparseSemanticEchoField
from sara_engine.language.semantic_events import LanguageEvent
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "role_labelled_heldout_cases_test_large.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_raw_integration_ablation.json")
TOKEN_RE = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _units(text: str, *, language: str, mode: str) -> Tuple[str, ...]:
    if mode == "surface" or language == "en":
        return tuple(token.casefold() for token in TOKEN_RE.findall(text))
    chars = tuple(char.casefold() for char in text if not char.isspace())
    if mode == "character":
        return chars
    if mode == "bigram":
        return chars if len(chars) < 2 else tuple("".join(chars[index : index + 2]) for index in range(len(chars) - 1))
    raise ValueError(mode)


def _run(case: Mapping[str, Any], *, mode: str, gap_cap: int) -> Dict[str, Any]:
    language = str(case["language"])
    document = _units(str(case["document"]), language=language, mode=mode)
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    endpoints = [str(edge["head"]), str(edge["dependent"])]
    endpoint_units = [_units(endpoint, language=language, mode=mode) for endpoint in endpoints]
    field = SparseSemanticEchoField(tiers=("fast", "medium", "slow"), max_echoes=9, max_comparisons=16, enable_role_binding=False)
    events = tuple((1, LanguageEvent(index, "orthographic", unit)) for index, unit in enumerate(document))
    query_units = tuple(unit for units in endpoint_units for unit in units)
    query = tuple((gap_cap if index == 0 else 1, LanguageEvent(index, "orthographic", unit)) for index, unit in enumerate(query_units))
    traces = field.run(events + query)
    decision_features = {str(decision.feature).casefold() for trace in traces for decision in trace.decisions}
    endpoint_hits = [all(unit in decision_features for unit in units) for units in endpoint_units]
    return {
        "case_id": str(case["case_id"]),
        "language": language,
        "task_family": str(case.get("task_family", "")),
        "correct": all(endpoint_hits),
        "any_endpoint": any(endpoint_hits),
        "state_bytes": field.serialized_state_bytes(),
        "active_echoes": max((trace.active_echoes for trace in traces), default=0),
        "comparisons": max((trace.comparisons for trace in traces), default=0),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def accuracy(items: Sequence[Mapping[str, Any]], field: str) -> float:
        return sum(bool(item[field]) for item in items) / max(1, len(items))

    languages: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        languages[str(row["language"])].append(row)
    state = max((int(row["state_bytes"]) for row in rows), default=0)
    return {
        "strict_both_endpoint_accuracy": accuracy(rows, "correct"),
        "any_endpoint_accuracy": accuracy(rows, "any_endpoint"),
        "strict_by_language": {key: accuracy(value, "correct") for key, value in sorted(languages.items())},
        "max_state_bytes": state,
        "max_active_echoes": max((int(row["active_echoes"]) for row in rows), default=0),
        "max_comparisons": max((int(row["comparisons"]) for row in rows), default=0),
        "bounded": state <= 4096 and max((int(row["active_echoes"]) for row in rows), default=0) <= 9 and max((int(row["comparisons"]) for row in rows), default=0) <= 16,
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    configs = {
        "surface_gap18": {"mode": "surface", "gap_cap": 18},
        "character_gap18": {"mode": "character", "gap_cap": 18},
        "bigram_gap18": {"mode": "bigram", "gap_cap": 18},
    }
    variants = {name: _summary([_run(case, **options) for case in cases]) for name, options in configs.items()}
    return {
        "schema": "sara-semantic-echo-ud-raw-integration-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "identity_aware": False,
        "interpretation": [
            "This integrates only raw-text-safe gap capping and tokenization variants.",
            "Observed UD token IDs are intentionally not used because they are unavailable from raw text alone.",
            "Strict scoring requires every unit of both query endpoints to reappear as an echo decision.",
            "No production promotion is claimed from this diagnostic ablation.",
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
