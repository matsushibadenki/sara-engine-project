#!/usr/bin/env python3
"""Evaluate fixed-size chunked echo memory for raw UD text."""

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
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_chunked_memory_ablation.json")
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


def _run(case: Mapping[str, Any], *, mode: str, chunk_size: int, max_chunks: int) -> Dict[str, Any]:
    language = str(case["language"])
    units = _units(str(case["document"]), language=language, mode=mode)
    chunks = [units[index : index + chunk_size] for index in range(0, len(units), chunk_size)][-max_chunks:]
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    endpoint_units = [_units(str(edge["head"]), language=language, mode=mode), _units(str(edge["dependent"]), language=language, mode=mode)]
    matched = False
    total_state = 0
    max_active = 0
    max_comparisons = 0
    for chunk_index, chunk in enumerate(chunks):
        field = SparseSemanticEchoField(tiers=("fast", "medium", "slow"), max_echoes=6, max_comparisons=12, enable_role_binding=False)
        document_events = tuple((1, LanguageEvent(index, "orthographic", unit)) for index, unit in enumerate(chunk))
        query_units = tuple(unit for endpoint in endpoint_units for unit in endpoint)
        query_events = tuple((18 if index == 0 else 1, LanguageEvent(index, "orthographic", unit)) for index, unit in enumerate(query_units))
        traces = field.run(document_events + query_events)
        decisions = {str(decision.feature).casefold() for trace in traces for decision in trace.decisions}
        matched = matched or all(all(unit in decisions for unit in endpoint) for endpoint in endpoint_units)
        total_state += field.serialized_state_bytes()
        max_active = max(max_active, max((trace.active_echoes for trace in traces), default=0))
        max_comparisons = max(max_comparisons, max((trace.comparisons for trace in traces), default=0))
    return {
        "case_id": str(case["case_id"]),
        "language": language,
        "task_family": str(case.get("task_family", "")),
        "correct": matched,
        "chunk_count": len(chunks),
        "state_bytes": total_state,
        "max_active_echoes": max_active,
        "max_comparisons": max_comparisons,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def accuracy(items: Sequence[Mapping[str, Any]]) -> float:
        return sum(bool(item["correct"]) for item in items) / max(1, len(items))
    languages: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        languages[str(row["language"])].append(row)
    state = max((int(row["state_bytes"]) for row in rows), default=0)
    return {
        "accuracy": accuracy(rows),
        "by_language": {key: accuracy(value) for key, value in sorted(languages.items())},
        "max_state_bytes": state,
        "max_active_echoes": max((int(row["max_active_echoes"]) for row in rows), default=0),
        "max_comparisons": max((int(row["max_comparisons"]) for row in rows), default=0),
        "bounded": state <= 4096 and max((int(row["max_active_echoes"]) for row in rows), default=0) <= 6 and max((int(row["max_comparisons"]) for row in rows), default=0) <= 12,
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    configs = {
        "surface_chunk16x2": {"mode": "surface", "chunk_size": 16, "max_chunks": 2},
        "surface_chunk16x4": {"mode": "surface", "chunk_size": 16, "max_chunks": 4},
        "character_chunk16x2": {"mode": "character", "chunk_size": 16, "max_chunks": 2},
        "bigram_chunk16x2": {"mode": "bigram", "chunk_size": 16, "max_chunks": 2},
    }
    variants = {name: _summary([_run(case, **options) for case in cases]) for name, options in configs.items()}
    return {
        "schema": "sara-semantic-echo-ud-chunked-memory-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "interpretation": [
            "Chunks are selected by fixed recency only; target edge positions are not used for routing.",
            "Each chunk is independently bounded and queried with the same raw-text endpoint query.",
            "This is an ablation and does not modify production memory behavior.",
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
