#!/usr/bin/env python3
"""Add coarse positional buckets to a fixed bigram signature memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "role_labelled_heldout_cases_test_large.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_signature_proximity_ablation.json")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _bigrams(text: str) -> Tuple[str, ...]:
    chars = tuple(char.casefold() for char in text if not char.isspace())
    return chars if len(chars) < 2 else tuple("".join(chars[index : index + 2]) for index in range(len(chars) - 1))


def _position(feature: str, bucket: int, *, bit_count: int, hashes: int) -> Tuple[int, ...]:
    digest = hashlib.blake2b(f"{feature}\x00{bucket}".encode("utf-8"), digest_size=16).digest()
    return tuple(int.from_bytes(digest[index * 4 : index * 4 + 4], "big") % bit_count for index in range(hashes))


def _run(case: Mapping[str, Any], *, bit_count: int, bucket_size: int, hashes: int) -> Dict[str, Any]:
    document = _bigrams(str(case["document"]))
    bits = 0
    for index, unit in enumerate(document):
        bucket = index // bucket_size
        for position in _position(unit, bucket, bit_count=bit_count, hashes=hashes):
            bits |= 1 << position
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    endpoints = [_bigrams(str(edge["head"])), _bigrams(str(edge["dependent"]))]
    bucket_count = (len(document) + bucket_size - 1) // bucket_size
    matching_buckets = []
    for bucket in range(bucket_count):
        matching = True
        for endpoint in endpoints:
            for unit in endpoint:
                if not all(bits & (1 << position) for position in _position(unit, bucket, bit_count=bit_count, hashes=hashes)):
                    matching = False
                    break
            if not matching:
                break
        if matching:
            matching_buckets.append(bucket)
    return {
        "case_id": str(case["case_id"]),
        "language": str(case["language"]),
        "task_family": str(case.get("task_family", "")),
        "correct": bool(matching_buckets),
        "matching_bucket_count": len(matching_buckets),
        "state_bytes": (bit_count + 7) // 8,
        "bucket_size": bucket_size,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def accuracy(items: Sequence[Mapping[str, Any]]) -> float:
        return sum(bool(item["correct"]) for item in items) / max(1, len(items))
    languages: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        languages[str(row["language"])].append(row)
    return {
        "strict_proximity_accuracy": accuracy(rows),
        "by_language": {key: accuracy(value) for key, value in sorted(languages.items())},
        "state_bytes": max((int(row["state_bytes"]) for row in rows), default=0),
        "bounded": max((int(row["state_bytes"]) for row in rows), default=0) <= 4096,
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    configs = {
        "bigram_bits256_bucket4": {"bit_count": 256, "bucket_size": 4, "hashes": 3},
        "bigram_bits256_bucket8": {"bit_count": 256, "bucket_size": 8, "hashes": 3},
        "bigram_bits512_bucket8": {"bit_count": 512, "bucket_size": 8, "hashes": 3},
    }
    variants = {name: _summary([_run(case, **options) for case in cases]) for name, options in configs.items()}
    return {
        "schema": "sara-semantic-echo-ud-signature-proximity-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "diagnostic_upper_bound": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "interpretation": [
            "The memory stores only hashed bigram-plus-position-bucket signatures.",
            "A strict hit requires all bigrams of both endpoints to be present in one coarse bucket.",
            "This is a compact proximity diagnostic; it does not preserve order or dependency role.",
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
