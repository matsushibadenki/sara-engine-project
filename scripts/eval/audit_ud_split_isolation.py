#!/usr/bin/env python3
"""Audit train/dev/test separation for collected UD role-labelled cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_PROCESSED_DIR = processed_data_path("phase19_20_language")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "audit_ud_split_isolation.json")


def _load(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(*, processed_dir: str = DEFAULT_PROCESSED_DIR) -> Dict[str, Any]:
    root = Path(processed_dir)
    paths = {
        "train": root / "role_labelled_training_cases_train.jsonl",
        "dev": root / "role_labelled_heldout_cases_dev_large.jsonl",
        "test": root / "role_labelled_heldout_cases_test_large.jsonl",
    }
    rows = {split: _load(path) for split, path in paths.items()}
    keys = {
        split: {(str(row.get("treebank")), str(row.get("source_sentence_id"))) for row in items}
        for split, items in rows.items()
    }
    content_hashes = {
        split: {str(row.get("source_hash")) for row in items}
        for split, items in rows.items()
    }
    overlaps = {
        f"{left}_{right}": {
            "sentence_keys": len(keys[left] & keys[right]),
            "source_hashes": len(content_hashes[left] & content_hashes[right]),
        }
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    eval_files = {"dev": str(paths["dev"].resolve()), "test": str(paths["test"].resolve())}
    train_referenced_by_eval = any("training_cases_train" in path for path in eval_files.values())
    return {
        "schema": "sara-ud-split-isolation-audit-v1",
        "processed_dir": str(root.resolve()),
        "files": {
            split: {
                "path": str(path.resolve()),
                "exists": path.exists(),
                "sha256": _digest(path) if path.exists() else None,
                "case_count": len(rows[split]),
            }
            for split, path in paths.items()
        },
        "overlaps": overlaps,
        "evaluation_files": eval_files,
        "train_referenced_by_eval": train_referenced_by_eval,
        "source_isolation_passed": all(value["sentence_keys"] == 0 and value["source_hashes"] == 0 for value in overlaps.values()),
        "evaluation_isolation_passed": not train_referenced_by_eval,
        "passed": all(value["sentence_keys"] == 0 and value["source_hashes"] == 0 for value in overlaps.values()) and not train_referenced_by_eval,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)
    report = build_report(processed_dir=args.processed_dir)
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"passed": report["passed"], "report_path": os.path.abspath(args.report_path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
