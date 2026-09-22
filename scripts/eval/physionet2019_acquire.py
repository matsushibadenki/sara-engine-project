#!/usr/bin/env python3
"""Acquire pinned official patient files without parsing clinical columns."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5, sha256
import json
import os
from pathlib import Path
import re
import tempfile

from sara_engine.utils.project_paths import (
    ensure_output_directory, ensure_parent_directory, processed_data_path,
    raw_data_path, workspace_path,
)
from physionet2019_s3_inventory import summarize_objects

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "physionet2019_acquisition_protocol_v1.json"))
INVENTORY_PROTOCOL = Path(processed_data_path("benchmark_fixtures", "physionet2019_s3_inventory_protocol_v1.json"))
INVENTORY = Path(workspace_path("evaluation", "physionet2019_s3_inventory_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "physionet2019_acquisition_v1.json"))
PROTOCOL_SHA256 = "ff96e8e12b40e06ba60790d32d7c0b3709ba6d0edfdf31b06da5801bd27e0ca5"


def file_hashes(path: Path) -> tuple[int, str, str]:
    length = 0
    digest_md5 = md5()
    digest_sha256 = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            length += len(block)
            digest_md5.update(block)
            digest_sha256.update(block)
    return length, digest_md5.hexdigest(), digest_sha256.hexdigest()


def download_one(client: object, *, bucket: str, key: str, size: int,
                 etag: str, prefix: str, scratch: Path) -> tuple[str, int, str, bool]:
    """Verify single-part ETag and size before atomically adding a raw file."""
    if not re.fullmatch(r"[0-9a-f]{32}", etag):
        raise ValueError("Source ETag is not a single-part MD5")
    relative = key.removeprefix(prefix)
    if relative == key:
        raise ValueError("Source key is outside frozen prefix")
    target = Path(ensure_parent_directory(raw_data_path("r2_physionet2019", relative)))
    if target.exists():
        actual_size, actual_md5, actual_sha256 = file_hashes(target)
        if (actual_size, actual_md5) != (size, etag):
            raise ValueError("Existing raw file does not match frozen source")
        return key, size, actual_sha256, True
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    temporary = None
    try:
        if response["ContentLength"] != size or response["ETag"].strip('"') != etag:
            raise ValueError("Source object changed during acquisition")
        with tempfile.NamedTemporaryFile(dir=scratch, prefix="patient-", delete=False) as stream:
            temporary = Path(stream.name)
            digest_md5 = md5()
            digest_sha256 = sha256()
            length = 0
            for block in body.iter_chunks(chunk_size=65536):
                if not block:
                    continue
                stream.write(block)
                length += len(block)
                digest_md5.update(block)
                digest_sha256.update(block)
        if length != size or digest_md5.hexdigest() != etag:
            raise ValueError("Downloaded body failed size or MD5 verification")
        os.link(temporary, target)
        return key, size, digest_sha256.hexdigest(), False
    finally:
        body.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="Acquire only this many files for an integrity smoke test; do not finalize")
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    if args.limit < 0 or not 1 <= args.workers <= 96:
        raise ValueError("Invalid acquisition limits")
    if OUTPUT.exists():
        raise ValueError("PhysioNet acquisition result already exists")
    if sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Acquisition protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(INVENTORY.read_bytes()).hexdigest() != protocol["source_inventory_sha256"]:
        raise ValueError("Pinned source inventory changed")
    inventory_protocol = json.loads(INVENTORY_PROTOCOL.read_text())

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED,
                           max_pool_connections=args.workers,
                           retries={"max_attempts": 5, "mode": "standard"}))
    pages = list(client.get_paginator("list_objects_v2").paginate(
        Bucket=protocol["bucket"], Prefix=protocol["prefix"]))
    summary = summarize_objects(pages, inventory_protocol)
    if (not summary["passed"] or
            summary["object_metadata_sha256"] != protocol["expected_object_metadata_sha256"] or
            summary["object_counts"] != protocol["expected_counts"] or
            sum(summary["total_bytes"].values()) != protocol["expected_total_bytes"]):
        raise ValueError("Current source inventory differs from frozen inventory")
    entries = sorted((entry["Key"], entry["Size"], entry["ETag"].strip('"'))
                     for page in pages for entry in page.get("Contents", []))
    if any(not re.fullmatch(r"[0-9a-f]{32}", etag) for _, _, etag in entries):
        raise ValueError("At least one source ETag is not a single-part MD5")
    if args.limit:
        entries = entries[:args.limit]
    scratch = Path(ensure_output_directory(workspace_path("evaluation", "physionet2019_parts")))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, client, bucket=protocol["bucket"],
                                   key=key, size=size, etag=etag, prefix=protocol["prefix"],
                                   scratch=scratch) for key, size, etag in entries]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 1000 == 0:
                print(json.dumps({"verified_files": len(results), "target_files": len(entries)}), flush=True)
    content_digest = sha256()
    for key, size, body_sha256, _ in sorted(results):
        content_digest.update(f"{key}\t{size}\t{body_sha256}\n".encode())
    result = {"schema": "sara-physionet2019-acquisition-result-v1",
              "protocol_sha256": PROTOCOL_SHA256,
              "source_inventory_sha256": protocol["source_inventory_sha256"],
              "source_object_metadata_sha256": summary["object_metadata_sha256"],
              "verified_files": len(results), "verified_bytes": sum(row[1] for row in results),
              "resumed_verified_files": sum(row[3] for row in results),
              "content_manifest_sha256": content_digest.hexdigest(),
              "complete": not args.limit, "clinical_columns_parsed": False,
              "patient_labels_interpreted": False, "candidate_scored": False}
    if args.limit:
        print(json.dumps(result, sort_keys=True))
        return 0
    if result["verified_files"] != sum(protocol["expected_counts"].values()) or result["verified_bytes"] != protocol["expected_total_bytes"]:
        raise ValueError("Acquisition count or byte gate failed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
