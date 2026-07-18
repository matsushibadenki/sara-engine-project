#!/usr/bin/env python3
"""Measure deterministic reservoir retention of exact bigram-bucket pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CASE_PATH = processed_data_path("phase19_20_language", "role_labelled_heldout_cases_test_large.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "semantic_echo_ud_exact_pair_reservoir_ablation.json")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _bigrams(text: str) -> Tuple[str, ...]:
    chars = tuple(char.casefold() for char in text if not char.isspace())
    return chars if len(chars) < 2 else tuple("".join(chars[index : index + 2]) for index in range(len(chars) - 1))


def _priority(unit: str, index: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{index}\x00{unit}".encode("utf-8"), digest_size=8).digest(), "big")


def _run(case: Mapping[str, Any], *, capacity: int, bucket_size: int) -> Dict[str, Any]:
    document = _bigrams(str(case["document"]))
    candidates = [( _priority(unit, index), unit, index // bucket_size) for index, unit in enumerate(document)]
    memory = sorted(candidates)[:capacity]
    retained = {(unit, bucket) for _, unit, bucket in memory}
    edge = max(case["dependency_or_role_edges"], key=lambda item: int(item.get("distance", 0)))
    endpoints = [_bigrams(str(edge["head"])), _bigrams(str(edge["dependent"]))]
    buckets = sorted({bucket for _, _, bucket in memory})
    matching = [bucket for bucket in buckets if all((unit, bucket) in retained for endpoint in endpoints for unit in endpoint)]
    state = {"schema": "sara-exact-pair-reservoir-v1", "entries": [(unit, bucket) for _, unit, bucket in memory]}
    state_bytes = len(json.dumps(state, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "case_id": str(case["case_id"]),
        "language": str(case["language"]),
        "task_family": str(case.get("task_family", "")),
        "correct": bool(matching),
        "matching_bucket_count": len(matching),
        "state_bytes": state_bytes,
        "active_entries": len(memory),
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
        "max_state_bytes": max((int(row["state_bytes"]) for row in rows), default=0),
        "max_active_entries": max((int(row["active_entries"]) for row in rows), default=0),
        "bounded": max((int(row["state_bytes"]) for row in rows), default=0) <= 4096,
    }


def build_report(*, case_path: str = DEFAULT_CASE_PATH) -> Dict[str, Any]:
    cases = _load(case_path)
    configs = {
        "reservoir32_bucket4": {"capacity": 32, "bucket_size": 4},
        "reservoir64_bucket4": {"capacity": 64, "bucket_size": 4},
        "reservoir128_bucket8": {"capacity": 128, "bucket_size": 8},
    }
    variants = {name: _summary([_run(case, **options) for case in cases]) for name, options in configs.items()}
    return {
        "schema": "sara-semantic-echo-ud-exact-pair-reservoir-ablation-v1",
        "phase": "19/20",
        "observed_only": True,
        "diagnostic_upper_bound": True,
        "source_isolation_preserved": all(str(case.get("evidence_scope")) == "independent_external" for case in cases),
        "case_path": os.path.abspath(case_path),
        "variants": variants,
        "interpretation": [
            "A deterministic priority reservoir removes fixed recency bias while retaining exact observed pairs.",
            "No target edge position or dependency role selects reservoir entries.",
            "This is a bounded lexical diagnostic and does not establish production role binding.",
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
