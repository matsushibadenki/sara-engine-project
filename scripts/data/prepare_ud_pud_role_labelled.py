#!/usr/bin/env python3
"""Prepare an isolated observed-only held-out set from UD PUD treebanks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from prepare_ud_role_labelled import RAW_DIR, COLLECTION_TIME, _hash, _read_sentences, _select, _edges


SOURCES = [
    {
        "language": "en",
        "treebank": "UD_English-PUD",
        "file": "en_pud-ud-test.conllu",
        "source_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-PUD/r2.18/en_pud-ud-test.conllu",
        "license": "CC BY-SA 3.0; data/raw/ud_role_labelled/UD_English-PUD.LICENSE",
    },
    {
        "language": "ja",
        "treebank": "UD_Japanese-PUD",
        "file": "ja_pud-ud-test.conllu",
        "source_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-PUD/r2.18/ja_pud-ud-test.conllu",
        "license": "CC BY-SA 3.0; data/raw/ud_role_labelled/UD_Japanese-PUD.LICENSE",
    },
    {
        "language": "zh-CN",
        "treebank": "UD_Chinese-PUD",
        "file": "zh_pud-ud-test.conllu",
        "source_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-PUD/r2.18/zh_pud-ud-test.conllu",
        "license": "CC BY-SA 3.0; data/raw/ud_role_labelled/UD_Chinese-PUD.LICENSE",
    },
]


def build(limit: int = 20) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source in SOURCES:
        source_path = RAW_DIR / source["file"]
        selected = _select(_read_sentences(source_path), limit=limit)
        source_bytes = source_path.read_bytes()
        source_file_hash = hashlib.sha256(source_bytes).hexdigest()
        for index, item in enumerate(selected):
            sentence = item["sentence"]
            content = str(sentence["text"])
            edges = item["edges"]
            rows.append(
                {
                    "schema": "sara-independent-role-labelled-case-v1",
                    "case_id": f"{source['treebank'].lower()}-test-{index:03d}",
                    "language": source["language"],
                    "treebank": source["treebank"],
                    "task_type": "structural",
                    "task_family": item["task_family"],
                    "query": "Retrieve the observed dependency edges and their head-dependent roles from this sentence.",
                    "document": content,
                    "dependency_or_role_edges": edges,
                    "source_url": source["source_url"],
                    "source_domain": "raw.githubusercontent.com",
                    "source_hash": _hash(content),
                    "source_file_hash": source_file_hash,
                    "source_revision": "UD v2.18 PUD test split; retrieved 2026-07-18",
                    "collection_time": COLLECTION_TIME,
                    "evidence_scope": "independent_external",
                    "observed_only": True,
                    "compliance_level": "allow",
                    "license_hint": source["license"],
                    "near_duplicate_signature": _hash(re.sub(r"\W+", " ", content.lower()))[:16],
                    "expected_behavior": "retrieve",
                    "derivation_stage": "post_source_split",
                    "source_sentence_id": sentence.get("sent_id", ""),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="data/processed/phase19_20_language/pud_role_labelled_heldout_cases_test.jsonl")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    rows = build(limit=args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Prepared {len(rows)} isolated PUD cases")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
