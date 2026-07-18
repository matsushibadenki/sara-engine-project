#!/usr/bin/env python3
"""Collect a small provenance-qualified architecture-migration pilot batch."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "architecture_migration" / "source_rows.jsonl"
PROCESSED_PATH = ROOT / "data" / "processed" / "autobot" / "architecture_migration_latent_manifest.jsonl"


COLLECTION_TIME = "2026-07-18T00:00:00Z"

# These are short canonical excerpts taken from the authoritative pages listed below.
# They are stored as observed source material; no generated answer or fixture text is used.
SOURCE_ROWS = [
    {
        "record_id": "arch-migration-python-001",
        "content": "The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv. The argparse module also automatically generates help and usage messages and issues errors when users give invalid arguments.",
        "source_url": "https://docs.python.org/3/library/argparse.html",
        "source_revision": "Python 3.14.6 documentation, 2026-07-14",
        "license_hint": "Python Documentation License 2.0; https://docs.python.org/3/license.html",
        "task_type": "migration",
    },
    {
        "record_id": "arch-migration-python-002",
        "content": "The ArgumentParser.add_argument method attaches individual argument specifications to the parser. It supports positional arguments, options that accept values, and on/off flags. The parse_args method runs the parser and places the extracted data in an argparse.Namespace object.",
        "source_url": "https://docs.python.org/3/library/argparse.html#the-add-argument-method",
        "source_revision": "Python 3.14.6 documentation, 2026-07-14",
        "license_hint": "Python Documentation License 2.0; https://docs.python.org/3/license.html",
        "task_type": "delayed",
    },
    {
        "record_id": "arch-migration-python-003",
        "content": "This module offers classes representing filesystem paths with semantics appropriate for different operating systems. Path classes are divided between pure paths, which provide purely computational operations without I/O, and concrete paths, which inherit from pure paths but also provide I/O operations.",
        "source_url": "https://docs.python.org/3/library/pathlib.html",
        "source_revision": "Python 3.14.6 documentation, 2026-07-14",
        "license_hint": "Python Documentation License 2.0; https://docs.python.org/3/license.html",
        "task_type": "migration",
    },
    {
        "record_id": "arch-migration-ietf-001",
        "content": "The Hypertext Transfer Protocol is a stateless application-level protocol for distributed, collaborative, hypertext information systems. RFC 9110 describes the overall architecture of HTTP, establishes common terminology, and defines aspects of the protocol shared by all versions.",
        "source_url": "https://www.ietf.org/rfc/rfc9110.html#abstract",
        "source_revision": "RFC 9110, June 2022",
        "license_hint": "IETF Trust Legal Provisions; https://trustee.ietf.org/license-info/",
        "task_type": "migration",
    },
    {
        "record_id": "arch-migration-ietf-002",
        "content": "HTTP provides a uniform interface for interacting with a resource regardless of its type, nature, or implementation by sending messages that manipulate or transfer representations. Each message is either a request or a response, and the client examines received responses to determine what to do next.",
        "source_url": "https://www.ietf.org/rfc/rfc9110.html#section-1.3",
        "source_revision": "RFC 9110, June 2022",
        "license_hint": "IETF Trust Legal Provisions; https://trustee.ietf.org/license-info/",
        "task_type": "delayed",
    },
    {
        "record_id": "arch-migration-ietf-003",
        "content": "The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, NOT RECOMMENDED, MAY, and OPTIONAL are to be interpreted as described in BCP 14 when, and only when, they appear in all capitals. An implementation is conformant if it complies with the requirements associated with the roles it partakes in.",
        "source_url": "https://www.ietf.org/rfc/rfc9110.html#section-2.2",
        "source_revision": "RFC 9110, June 2022",
        "license_hint": "IETF Trust Legal Provisions; https://trustee.ietf.org/license-info/",
        "task_type": "migration",
    },
]


def _signature(content: str, width: int = 4096) -> list[int]:
    tokens = re.findall(r"[a-z0-9]+", content.lower())
    positions = {int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % width for token in tokens}
    return sorted(positions)


def build_rows() -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for index, item in enumerate(SOURCE_ROWS):
        content = item["content"]
        source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        near_duplicate = hashlib.sha256(re.sub(r"\W+", " ", content.lower()).encode("utf-8")).hexdigest()[:16]
        raw = {
            "schema": "sara-independent-source-row-v1",
            **item,
            "source_domain": item["source_url"].split("//", 1)[1].split("/", 1)[0].lower(),
            "source_hash": source_hash,
            "collection_time": COLLECTION_TIME,
            "evidence_scope": "independent_external",
            "observed_only": True,
            "compliance_level": "allow",
            "near_duplicate_signature": near_duplicate,
            "content_origin": "transcribed_source_excerpt",
        }
        raw_rows.append(raw)
        manifest_rows.append(
            {
                "schema": "sara-own-latent-manifest-row-v1",
                "manifest_id": f"architecture_migration_latent_{index:06d}",
                "material_hash": source_hash,
                "source_url": raw["source_url"],
                "source_domain": raw["source_domain"],
                "source_ref": raw["source_url"],
                "source_type": "independent_external_documentation",
                "source_revision": raw["source_revision"],
                "license_hint": raw["license_hint"],
                "observed_only": True,
                "compliance_level": "allow",
                "quality_score": 1.0,
                "event_cost": len(_signature(content)),
                "language": "en",
                "material_type": "source_claim",
                "latent_cluster_id": f"external_{index:06d}",
                "sparse_signature": _signature(content),
                "evidence_scope": "independent_external",
                "collection_time": COLLECTION_TIME,
                "migration_horizon_index": index,
            }
        )
    return raw_rows, manifest_rows


def main() -> int:
    raw_rows, manifest_rows = build_rows()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with PROCESSED_PATH.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Collected {len(raw_rows)} observed-only source rows")
    print(f"Raw output: {RAW_PATH}")
    print(f"Processed output: {PROCESSED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
