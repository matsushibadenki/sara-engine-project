#!/usr/bin/env python3
"""Measure fixed-bit lexical signature memory for raw UD text coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "role_labelled_heldout_cases_test_large.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_signature_memory_ablation.json")
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


def _positions(feature: str, *, bit_count: int, hashes: int) -> Tuple[int, ...]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
    return tuple(int.from_bytes(digest[index * 4 : index * 4 + 4], "big") % bit_count for index in range(hashes))


def _run(case: Mapping[str, Any], *, mode: str, bit_count: int, hashes: int) -> Dict[str, Any]:
    language = str(case["language"])
    bits = 0
    for unit in _units(str(case["document"]), language=language, mode=mode):
        for position in _positions(unit, bit_count=bit_count, hashes=hashes):
            bits |= 1 << position
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    endpoints = [_units(str(edge["head"]), language=language, mode=mode), _units(str(edge["dependent"]), language=language, mode=mode)]
    endpoint_hits = [all(all(bits & (1 << position) for position in _positions(unit, bit_count=bit_count, hashes=hashes)) for unit in endpoint) for endpoint in endpoints]
    state_bytes = (bit_count + 7) // 8
    return {
        "case_id": str(case["case_id"]),
        "language": language,
        "task_family": str(case.get("task_family", "")),
        "correct": all(endpoint_hits),
        "any_endpoint": any(endpoint_hits),
        "state_bytes": state_bytes,
        "bit_count": bit_count,
        "hashes": hashes,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def accuracy(items: Sequence[Mapping[str, Any]], field: str) -> float:
        return sum(bool(item[field]) for item in items) / max(1, len(items))
    languages: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        languages[str(row["language"])].append(row)
    return {
        "strict_both_endpoint_accuracy": accuracy(rows, "correct"),
        "any_endpoint_accuracy": accuracy(rows, "any_endpoint"),
        "by_language": {key: accuracy(value, "correct") for key, value in sorted(languages.items())},
        "state_bytes": max((int(row["state_bytes"]) for row in rows), default=0),
        "bounded": max((int(row["state_bytes"]) for row in rows), default=0) <= 4096,
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    configs = {
        "surface_bits256_h3": {"mode": "surface", "bit_count": 256, "hashes": 3},
        "character_bits256_h3": {"mode": "character", "bit_count": 256, "hashes": 3},
        "bigram_bits256_h3": {"mode": "bigram", "bit_count": 256, "hashes": 3},
        "bigram_bits512_h3": {"mode": "bigram", "bit_count": 512, "hashes": 3},
    }
    variants = {name: _summary([_run(case, **options) for case in cases]) for name, options in configs.items()}
    return {
        "schema": "sara-semantic-echo-ud-signature-memory-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "diagnostic_upper_bound": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "interpretation": [
            "The fixed bitset stores lexical presence only and does not retain order, distance, or dependency role.",
            "Strict scoring requires all units of both endpoints to pass signature membership.",
            "This is a compact coverage diagnostic, not a production role-binding implementation.",
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
