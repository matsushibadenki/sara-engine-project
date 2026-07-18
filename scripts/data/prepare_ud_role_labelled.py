#!/usr/bin/env python3
"""Prepare a small role-labelled held-out set from official UD test splits."""

from __future__ import annotations

import hashlib
import json
import re
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "ud_role_labelled"
OUTPUT = ROOT / "data" / "processed" / "phase19_20_language" / "role_labelled_heldout_cases.jsonl"
COLLECTION_TIME = "2026-07-18T00:00:00Z"

SOURCES = [
    {
        "language": "en",
        "treebank": "UD_English-EWT",
        "file_template": "en_ewt-ud-{split}.conllu",
        "url_template": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-{split}.conllu",
        "license": "CC BY-SA 4.0; data/raw/ud_role_labelled/UD_English-EWT.LICENSE",
    },
    {
        "language": "ja",
        "treebank": "UD_Japanese-GSD",
        "file_template": "ja_gsd-ud-{split}.conllu",
        "url_template": "https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-GSD/master/ja_gsd-ud-{split}.conllu",
        "license": "CC BY-SA 4.0; data/raw/ud_role_labelled/UD_Japanese-GSD.LICENSE",
    },
    {
        "language": "zh-CN",
        "treebank": "UD_Chinese-GSDSimp",
        "file_template": "zh_gsdsimp-ud-{split}.conllu",
        "url_template": "https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSDSimp/master/zh_gsdsimp-ud-{split}.conllu",
        "license": "CC BY-SA 4.0; data/raw/ud_role_labelled/UD_Chinese-GSDSimp.LICENSE",
    },
]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_sentences(path: Path) -> List[Dict[str, Any]]:
    sentences: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {"tokens": [], "comments": []}
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if line.startswith("#"):
            current["comments"].append(line)
            if line.startswith("# sent_id = "):
                current["sent_id"] = line.split("=", 1)[1].strip()
            elif line.startswith("# text = "):
                current["text"] = line.split("=", 1)[1].strip()
            continue
        if line.strip():
            fields = line.split("\t")
            if "-" not in fields[0] and "." not in fields[0] and len(fields) >= 8:
                current["tokens"].append(
                    {
                        "id": int(fields[0]),
                        "form": fields[1],
                        "lemma": fields[2],
                        "upos": fields[3],
                        "head": int(fields[6]) if fields[6].isdigit() else 0,
                        "deprel": fields[7],
                    }
                )
            continue
        if current.get("tokens") and current.get("text"):
            sentences.append(current)
        current = {"tokens": [], "comments": []}
    return sentences


def _edges(sentence: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_id = {token["id"]: token for token in sentence["tokens"]}
    result = []
    for token in sentence["tokens"]:
        head = by_id.get(token["head"])
        if head is None or token["deprel"] in {"punct", "root"}:
            continue
        result.append(
            {
                "dependent": token["form"],
                "dependent_id": token["id"],
                "head": head["form"],
                "head_id": head["id"],
                "relation": token["deprel"],
                "distance": abs(token["id"] - head["id"]),
            }
        )
    return result


def _task_family(sentence: Dict[str, Any], edges: List[Dict[str, Any]]) -> str:
    relations = {edge["relation"] for edge in edges}
    if "neg" in relations or any(token["lemma"] in {"not", "n't", "ない", "無い", "不"} for token in sentence["tokens"]):
        return "negation_scope"
    if "nsubj" in relations and ("obj" in relations or "iobj" in relations):
        return "role_binding"
    if any(edge["relation"].split(":", 1)[0] in {"acl", "advcl", "ccomp", "xcomp"} and edge["distance"] >= 6 for edge in edges):
        return "long_distance_dependency"
    if any(token["upos"] == "PRON" for token in sentence["tokens"]):
        return "anaphora"
    return "structural_dependency"


def _select(sentences: Iterable[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    candidates = []
    for sentence in sentences:
        edges = _edges(sentence)
        if not edges:
            continue
        max_distance = max(edge["distance"] for edge in edges)
        relation_count = len({edge["relation"] for edge in edges})
        candidates.append((max_distance, relation_count, len(sentence["tokens"]), sentence, edges, _task_family(sentence, edges)))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    selected: List[Dict[str, Any]] = []
    selected_families = set()
    for _, _, _, sentence, edges, family in candidates:
        if len(selected) >= limit:
            break
        if family in selected_families:
            continue
        selected.append({"sentence": sentence, "edges": edges, "task_family": family})
        selected_families.add(family)
    for _, _, _, sentence, edges, family in candidates:
        if len(selected) >= limit:
            break
        if any(item["sentence"].get("sent_id") == sentence.get("sent_id") for item in selected):
            continue
        selected.append({"sentence": sentence, "edges": edges, "task_family": family})
    return selected


def build(split: str, limit: int = 5) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source in SOURCES:
        source_path = RAW_DIR / source["file_template"].format(split=split)
        selected = _select(_read_sentences(source_path), limit=limit)
        source_revision = f"UD v2.18 treebank {split} split; retrieved 2026-07-18"
        for index, item in enumerate(selected):
            sentence = item["sentence"]
            edges = item["edges"]
            content = str(sentence["text"])
            source_hash = _hash(content)
            task_family = item["task_family"]
            rows.append(
                {
                    "schema": "sara-independent-role-labelled-case-v1",
                    "case_id": f"{source['treebank'].lower()}-{split}-{index:03d}",
                    "language": source["language"],
                    "treebank": source["treebank"],
                    "task_type": "structural",
                    "task_family": task_family,
                    "query": "Retrieve the observed dependency edges and their head-dependent roles from this sentence.",
                    "document": content,
                    "dependency_or_role_edges": edges,
                    "source_url": source["url_template"].format(split=split),
                    "source_domain": "raw.githubusercontent.com",
                    "source_hash": source_hash,
                    "source_revision": source_revision,
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
    parser.add_argument("--split", choices=("train", "test", "dev"), default="test")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    rows = build(args.split, limit=args.limit)
    default_name = f"role_labelled_training_cases_{args.split}.jsonl" if args.split == "train" else f"role_labelled_heldout_cases_{args.split}.jsonl"
    output = Path(args.output) if args.output else OUTPUT.with_name(default_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Prepared {len(rows)} role-labelled {args.split} cases")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
