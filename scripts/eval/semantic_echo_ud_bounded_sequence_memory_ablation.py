#!/usr/bin/env python3
"""Evaluate bounded ordered sequence memory with strict endpoint matching."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "role_labelled_heldout_cases_test_large.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_bounded_sequence_memory_ablation.json")
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
        return chars if len(chars) < 2 else tuple(chars[index : index + 2] for index in range(len(chars) - 1))
    raise ValueError(mode)


def _spans(window: Sequence[str], pattern: Sequence[str]) -> List[Tuple[int, int]]:
    if not pattern or len(pattern) > len(window):
        return []
    size = len(pattern)
    return [(index, index + size - 1) for index in range(len(window) - size + 1) if tuple(window[index : index + size]) == tuple(pattern)]


def _run(case: Mapping[str, Any], *, mode: str, window_size: int, max_endpoint_gap: int) -> Dict[str, Any]:
    language = str(case["language"])
    document = _units(str(case["document"]), language=language, mode=mode)
    window = document[-window_size:]
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    head_spans = _spans(window, _units(str(edge["head"]), language=language, mode=mode))
    dependent_spans = _spans(window, _units(str(edge["dependent"]), language=language, mode=mode))
    separations = [max(0, max(h[0], d[0]) - min(h[1], d[1]) - 1) for h in head_spans for d in dependent_spans]
    matched = bool(separations) and min(separations) <= max_endpoint_gap
    state = {"schema": "sara-sequence-memory-v1", "window": list(window)}
    state_bytes = len(json.dumps(state, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "case_id": str(case["case_id"]),
        "language": language,
        "task_family": str(case.get("task_family", "")),
        "correct": matched,
        "min_endpoint_gap": min(separations) if separations else None,
        "state_bytes": state_bytes,
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
        "max_state_bytes": max((int(row["state_bytes"]) for row in rows), default=0),
        "bounded": max((int(row["state_bytes"]) for row in rows), default=0) <= 4096,
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    configs = {
        "surface_window32_gap4": {"mode": "surface", "window_size": 32, "max_endpoint_gap": 4},
        "surface_window32_gap8": {"mode": "surface", "window_size": 32, "max_endpoint_gap": 8},
        "character_window32_gap8": {"mode": "character", "window_size": 32, "max_endpoint_gap": 8},
        "bigram_window32_gap8": {"mode": "bigram", "window_size": 32, "max_endpoint_gap": 8},
    }
    variants = {name: _summary([_run(case, **options) for case in cases]) for name, options in configs.items()}
    return {
        "schema": "sara-semantic-echo-ud-bounded-sequence-memory-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "diagnostic_upper_bound": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "interpretation": [
            "Only the bounded recent sequence is retained; no gold token IDs or gold edge positions select memory entries.",
            "Both endpoint surface sequences must occur in the retained window and satisfy the configured proximity bound.",
            "This is still an observed lexical diagnostic and does not establish role-binding quality or production readiness.",
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
