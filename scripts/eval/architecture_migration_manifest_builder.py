#!/usr/bin/env python3
"""Build a provenance-qualified architecture-migration manifest from latent records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402


DEFAULT_INPUT_PATH = processed_data_path("autobot", "latent_manifest.jsonl")
DEFAULT_OUTPUT_PATH = processed_data_path("autobot", "architecture_migration_external_manifest.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "architecture_migration_manifest_builder.json")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [payload for line in handle if line.strip() and isinstance((payload := json.loads(line)), dict)]


def build_manifest(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    qualified = []
    for row in rows:
        source_url = str(row.get("source_url", "") or "")
        domain = str(urlparse(source_url).hostname or "").lower()
        if (
            str(row.get("schema", "")) != "sara-own-latent-manifest-row-v1"
            or not source_url.startswith("https://")
            or domain == "example.org"
            or not str(row.get("material_hash", ""))
            or not isinstance(row.get("sparse_signature"), list)
        ):
            continue
        qualified.append({**row, "source_domain": domain})
    qualified.sort(key=lambda row: (str(row["source_domain"]), str(row["source_url"]), str(row["material_hash"])))
    domain_indices: Dict[str, int] = defaultdict(int)
    manifest = []
    for row in qualified:
        domain = str(row["source_domain"])
        horizon_index = domain_indices[domain]
        domain_indices[domain] += 1
        manifest.append({
            **row,
            "schema": "sara-architecture-migration-source-row-v1",
            "source_site": ".".join(domain.split(".")[-2:]),
            "migration_horizon_index": horizon_index,
            "provenance_digest": hashlib.sha256(
                f"{row['source_url']}|{row['material_hash']}|{row.get('manifest_id', '')}".encode("utf-8")
            ).hexdigest(),
        })
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)
    manifest = build_manifest(_read_jsonl(args.input_path))
    with open(ensure_parent_directory(args.output_path), "w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {"schema": "sara-architecture-migration-manifest-builder-v1", "qualified_count": len(manifest), "output_path": os.path.abspath(args.output_path)}
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
