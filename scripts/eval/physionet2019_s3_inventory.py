#!/usr/bin/env python3
"""Inventory official PhysioNet patient objects without opening their bodies."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.utils.project_paths import processed_data_path, workspace_path

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "physionet2019_s3_inventory_protocol_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "physionet2019_s3_inventory_v1.json"))


def summarize_objects(pages: object, protocol: dict) -> dict:
    """Require exact set paths and derive a stable object-metadata digest."""
    prefix = protocol["prefix"]
    expected_sets = protocol["expected_sets"]
    suffix = protocol["allowed_object_suffix"]
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    seen: set[str] = set()
    objects: list[tuple[str, int, str]] = []
    for page in pages:
        for entry in page.get("Contents", []):
            key = entry["Key"]
            relative = key.removeprefix(prefix) if key.startswith(prefix) else ""
            parts = relative.split("/")
            if (len(parts) != 2 or parts[0] not in expected_sets or
                    not parts[1].endswith(suffix) or not parts[1].startswith("p") or
                    not parts[1][1:-len(suffix)].isdigit()):
                raise ValueError("Unexpected source object path")
            if key in seen:
                raise ValueError("Duplicate source object")
            seen.add(key)
            size = entry["Size"]
            etag = entry["ETag"].strip('"')
            if not isinstance(size, int) or size <= 0 or not etag:
                raise ValueError("Source object lacks stable metadata")
            counts[parts[0]] += 1
            sizes[parts[0]] += size
            objects.append((key, size, etag))
    digest = sha256()
    for key, size, etag in sorted(objects):
        digest.update(f"{key}\t{size}\t{etag}\n".encode())
    checks = {name: counts[name] == required for name, required in expected_sets.items()}
    return {"schema": "sara-physionet2019-s3-inventory-result-v1",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()).hexdigest(),
            "object_metadata_sha256": digest.hexdigest(),
            "object_counts": dict(sorted(counts.items())),
            "total_bytes": dict(sorted(sizes.items())),
            "checks": checks, "passed": all(checks.values()),
            "object_bodies_read": False, "patient_labels_read": False,
            "descriptive_values_read": False, "candidate_scored": False}


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("PhysioNet inventory already exists")
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    protocol = json.loads(PROTOCOL.read_text())
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    pages = client.get_paginator("list_objects_v2").paginate(
        Bucket=protocol["bucket"], Prefix=protocol["prefix"])
    result = summarize_objects(pages, protocol)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
