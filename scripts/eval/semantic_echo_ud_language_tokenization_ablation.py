#!/usr/bin/env python3
"""Compare language-aware surface tokenization for bounded anchor memory."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.language.semantic_events import LanguageEvent
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "role_labelled_heldout_cases_test_large.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_language_tokenization_ablation.json")
_TOKEN_RE = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tokens(text: str, *, language: str, mode: str) -> Tuple[str, ...]:
    if mode == "surface" or language == "en":
        return tuple(token.casefold() for token in _TOKEN_RE.findall(text))
    chars = tuple(char.casefold() for char in text if not char.isspace())
    if mode == "character":
        return chars
    if mode == "bigram":
        return chars if len(chars) < 2 else tuple(chars[index : index + 2] for index in range(len(chars) - 1))
    raise ValueError(f"unknown tokenization mode: {mode}")


def _events(text: str, *, language: str, mode: str) -> Tuple[LanguageEvent, ...]:
    return tuple(LanguageEvent(time=index, axis="orthographic", feature=token) for index, token in enumerate(_tokens(text, language=language, mode=mode)))


def _run(case: Mapping[str, Any], *, mode: str, capacity: int) -> Dict[str, Any]:
    language = str(case["language"])
    document = _events(str(case["document"]), language=language, mode=mode)
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    expected = {str(edge["head"]).casefold(), str(edge["dependent"]).casefold()}
    anchors: Dict[str, int] = {}
    for index, event in enumerate(document, start=1):
        anchors[event.feature] = index
        if len(anchors) > capacity:
            oldest = min(anchors.items(), key=lambda item: (item[1], item[0]))[0]
            del anchors[oldest]
    # Match the gold surface forms against the same tokenizer's atomic units.
    query_units = set(_tokens(" ".join(sorted(expected)), language=language, mode=mode))
    hits = query_units & set(anchors)
    state = {"schema": "sara-language-anchor-memory-v1", "anchors": sorted(anchors.items())}
    state_bytes = len(json.dumps(state, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "case_id": str(case["case_id"]),
        "language": language,
        "task_family": str(case.get("task_family", "")),
        "correct": bool(hits),
        "capacity": capacity,
        "state_bytes": state_bytes,
        "active_anchors": len(anchors),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def accuracy(items: Sequence[Mapping[str, Any]]) -> float:
        return sum(bool(item["correct"]) for item in items) / max(1, len(items))

    languages: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    families: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        languages[str(row["language"])].append(row)
        families[str(row["task_family"])].append(row)
    max_state = max((int(row["state_bytes"]) for row in rows), default=0)
    return {
        "accuracy": accuracy(rows),
        "by_language": {key: accuracy(value) for key, value in sorted(languages.items())},
        "by_family": {key: accuracy(value) for key, value in sorted(families.items())},
        "max_state_bytes": max_state,
        "bounded": max_state <= 4096,
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    configs = {
        "surface_capacity32": {"mode": "surface", "capacity": 32},
        "character_capacity32": {"mode": "character", "capacity": 32},
        "bigram_capacity32": {"mode": "bigram", "capacity": 32},
    }
    variants = {name: _summary([_run(case, **options) for case in cases]) for name, options in configs.items()}
    return {
        "schema": "sara-semantic-echo-ud-language-tokenization-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "diagnostic_upper_bound": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "interpretation": [
            "Tokenization is changed only for this diagnostic ablation; production tokenization is unchanged.",
            "The cache remains a surface lexical upper bound and does not infer dependency roles.",
            "Cross-language accuracy is reported separately because character and bigram units are not comparable to English word units.",
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
