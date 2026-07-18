#!/usr/bin/env python3
"""Apply stricter endpoint coverage checks to language-aware anchor memory."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "role_labelled_heldout_cases_test_large.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_language_tokenization_precision_ablation.json")
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


def _run(case: Mapping[str, Any], *, mode: str, capacity: int) -> Dict[str, Any]:
    language = str(case["language"])
    document_units = _units(str(case["document"]), language=language, mode=mode)
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    endpoints = [str(edge["head"]), str(edge["dependent"])]
    anchors: Dict[str, int] = {}
    for index, unit in enumerate(document_units, start=1):
        anchors[unit] = index
        if len(anchors) > capacity:
            oldest = min(anchors.items(), key=lambda item: (item[1], item[0]))[0]
            del anchors[oldest]
    endpoint_coverage = [all(unit in anchors for unit in _units(endpoint, language=language, mode=mode)) for endpoint in endpoints]
    state = {"schema": "sara-language-anchor-memory-v2", "anchors": sorted(anchors.items())}
    state_bytes = len(json.dumps(state, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "case_id": str(case["case_id"]),
        "language": language,
        "task_family": str(case.get("task_family", "")),
        "any_endpoint": any(endpoint_coverage),
        "both_endpoints": all(endpoint_coverage),
        "state_bytes": state_bytes,
    }


def _accuracy(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / max(1, len(rows))


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    languages: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        languages[str(row["language"])].append(row)
    return {
        "any_endpoint_accuracy": _accuracy(rows, "any_endpoint"),
        "both_endpoints_accuracy": _accuracy(rows, "both_endpoints"),
        "any_by_language": {key: _accuracy(value, "any_endpoint") for key, value in sorted(languages.items())},
        "both_by_language": {key: _accuracy(value, "both_endpoints") for key, value in sorted(languages.items())},
        "max_state_bytes": max((int(row["state_bytes"]) for row in rows), default=0),
        "bounded": max((int(row["state_bytes"]) for row in rows), default=0) <= 4096,
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
        "schema": "sara-semantic-echo-ud-language-tokenization-precision-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "diagnostic_upper_bound": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "interpretation": [
            "Any-endpoint matching is retained only as a comparison to the previous diagnostic.",
            "Both-endpoint matching requires all units of both labelled surface endpoints to remain in bounded memory.",
            "This remains lexical coverage, not dependency-role inference or production evidence.",
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
