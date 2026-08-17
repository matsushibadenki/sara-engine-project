#!/usr/bin/env python3
"""Collect bounded first-party documents for independent horizon evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    raw_data_path,
    workspace_path,
)


DEFAULT_RAW_PATH = raw_data_path("architecture_migration", "source_rows.jsonl")
DEFAULT_MANIFEST_PATH = processed_data_path(
    "autobot", "architecture_migration_latent_manifest.jsonl"
)
DEFAULT_REPORT_PATH = workspace_path(
    "evaluation", "continual_horizon_external_collection.json"
)
MAX_RESPONSE_BYTES = 2_000_000
MAX_CONTENT_CHARS = 12_000
ALLOWED_DOMAINS = frozenset({"docs.python.org", "www.rfc-editor.org"})


# Catalog stages are explicit so a horizon increase cannot silently widen collection.
SOURCE_CATALOG = (
    *(
        {
            "source_url": f"https://docs.python.org/3.14/library/{module}.html",
            "source_revision_hint": "Python 3.14 documentation",
            "license_hint": "Python Documentation License 2.0; https://docs.python.org/3/license.html",
            "catalog_stage": "horizon_10_pilot",
        }
        for module in (
            "json",
            "dataclasses",
            "asyncio",
            "typing",
            "sqlite3",
            "logging",
            "hashlib",
            "urllib.request",
        )
    ),
    *(
        {
            "source_url": f"https://www.rfc-editor.org/rfc/rfc{number}.html",
            "source_revision_hint": f"RFC {number}",
            "license_hint": "IETF Trust Legal Provisions; https://trustee.ietf.org/license-info/",
            "catalog_stage": "horizon_10_pilot",
        }
        for number in (9000, 9001, 9002, 9111, 9112, 9113, 9114, 9292)
    ),
    *(
        {
            "source_url": f"https://docs.python.org/3.14/library/{module}.html",
            "source_revision_hint": "Python 3.14 documentation",
            "license_hint": "Python Documentation License 2.0; https://docs.python.org/3/license.html",
            "catalog_stage": "horizon_30_expansion",
        }
        for module in (
            "collections",
            "functools",
            "itertools",
            "os",
            "sys",
            "time",
            "datetime",
            "zoneinfo",
            "re",
            "csv",
            "configparser",
            "tomllib",
            "decimal",
            "fractions",
            "statistics",
            "random",
            "secrets",
            "uuid",
            "concurrent.futures",
            "contextlib",
        )
    ),
    *(
        {
            "source_url": f"https://www.rfc-editor.org/rfc/rfc{number}.html",
            "source_revision_hint": f"RFC {number}",
            "license_hint": "IETF Trust Legal Provisions; https://trustee.ietf.org/license-info/",
            "catalog_stage": "horizon_30_expansion",
        }
        for number in (
            3986,
            5321,
            5322,
            6265,
            6455,
            6749,
            6750,
            7540,
            8200,
            8446,
            8941,
            8949,
            9204,
            9205,
            9234,
            9293,
            9330,
            9331,
            9332,
            9562,
        )
    ),
    *(
        {
            "source_url": f"https://docs.python.org/3.14/library/{module}.html",
            "source_revision_hint": "Python 3.14 documentation",
            "license_hint": "Python Documentation License 2.0; https://docs.python.org/3/license.html",
            "catalog_stage": "horizon_100_expansion",
        }
        for module in (
            "abc",
            "atexit",
            "builtins",
            "bisect",
            "calendar",
            "cmath",
            "code",
            "codecs",
            "codeop",
            "colorsys",
            "copy",
            "copyreg",
            "dbm",
            "difflib",
            "dis",
            "enum",
            "errno",
            "faulthandler",
            "fnmatch",
            "gc",
            "getopt",
            "getpass",
            "gettext",
            "glob",
            "graphlib",
            "gzip",
            "heapq",
            "hmac",
            "html",
            "http",
            "imaplib",
            "importlib",
            "inspect",
            "io",
            "ipaddress",
            "keyword",
            "linecache",
            "locale",
            "lzma",
            "mailbox",
            "marshal",
            "math",
            "mimetypes",
            "mmap",
            "netrc",
            "numbers",
            "operator",
            "pickle",
            "pickletools",
            "pkgutil",
            "platform",
            "plistlib",
            "pprint",
            "profile",
            "tarfile",
            "pty",
            "py_compile",
            "pyclbr",
            "queue",
            "quopri",
            "reprlib",
            "resource",
            "sched",
            "selectors",
            "shelve",
            "shlex",
            "shutil",
            "signal",
            "site",
            "smtplib",
        )
    ),
    *(
        {
            "source_url": f"https://www.rfc-editor.org/rfc/rfc{number}.html",
            "source_revision_hint": f"RFC {number}",
            "license_hint": "IETF Trust Legal Provisions; https://trustee.ietf.org/license-info/",
            "catalog_stage": "horizon_100_expansion",
        }
        for number in (
            2119,
            2616,
            3339,
            3552,
            3629,
            9147,
            4122,
            4648,
            4861,
            4949,
            5246,
            5280,
            5789,
            5869,
            5890,
            5952,
            6120,
            6234,
            6585,
            6761,
            6960,
            7001,
            7159,
            7230,
            7231,
            7232,
            7233,
            7234,
            7235,
            7240,
            7252,
            7303,
            7469,
            7519,
            7525,
            7538,
            7595,
            7644,
            7662,
            7725,
            7766,
            7807,
            7838,
            7854,
            7919,
            7924,
            7986,
            8032,
            8058,
            8081,
            8126,
            8141,
            8174,
            8259,
            8288,
            8336,
            8414,
            8484,
            8615,
            8620,
            8659,
            8705,
            8740,
            8785,
            8812,
            8879,
            8910,
            8996,
            9051,
            9106,
        )
    ),
)


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._suppressed_depth = 0
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        del attrs
        if tag.lower() in {"script", "style", "svg", "noscript"}:
            self._suppressed_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "svg", "noscript"} and self._suppressed_depth:
            self._suppressed_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._suppressed_depth:
            self.parts.append(data)


def _normalize_content(payload: bytes, content_type: str, charset: str) -> str:
    decoded = payload.decode(charset or "utf-8", errors="replace")
    if "html" in content_type.lower() or "<html" in decoded[:1000].lower():
        parser = _VisibleTextParser()
        parser.feed(decoded)
        decoded = " ".join(parser.parts)
    normalized = re.sub(r"\s+", " ", decoded).strip()
    if len(normalized) < 200:
        raise ValueError("source content is too short to qualify")
    return normalized[:MAX_CONTENT_CHARS]


def _signature(content: str, width: int = 4096) -> List[int]:
    tokens = re.findall(r"[a-z0-9]+", content.lower())
    return sorted(
        {
            int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % width
            for token in tokens
        }
    )


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [
            value
            for line in handle
            if line.strip() and isinstance((value := json.loads(line)), dict)
        ]


def _domain(value: Mapping[str, Any]) -> str:
    return str(
        value.get("source_domain")
        or urlparse(str(value.get("source_url", ""))).hostname
        or ""
    ).lower()


def select_needed_sources(
    existing_rows: Sequence[Mapping[str, Any]],
    target_horizon: int,
    catalog: Sequence[Mapping[str, str]] = SOURCE_CATALOG,
) -> List[Mapping[str, str]]:
    if target_horizon < 0:
        raise ValueError("target horizon must be non-negative")
    existing_refs = {
        str(row.get("source_ref") or row.get("source_url") or "")
        for row in existing_rows
    }
    counts = defaultdict(int)
    for row in existing_rows:
        if _domain(row) in ALLOWED_DOMAINS:
            counts[_domain(row)] += 1
    selected: List[Mapping[str, str]] = []
    for entry in catalog:
        domain = _domain(entry)
        source_url = str(entry["source_url"])
        if counts[domain] > target_horizon or source_url in existing_refs:
            continue
        selected.append(entry)
        existing_refs.add(source_url)
        counts[domain] += 1
    missing = {
        domain: target_horizon + 1 - counts[domain]
        for domain in sorted(ALLOWED_DOMAINS)
        if counts[domain] <= target_horizon
    }
    if missing:
        raise ValueError(f"reviewed source catalog cannot reach requested horizon: {missing}")
    return selected


def fetch_source(
    entry: Mapping[str, str], collection_time: str, timeout_seconds: float = 20.0
) -> Dict[str, Any]:
    source_url = str(entry["source_url"])
    requested_domain = str(urlparse(source_url).hostname or "").lower()
    if urlparse(source_url).scheme != "https" or requested_domain not in ALLOWED_DOMAINS:
        raise ValueError(f"source URL is outside the reviewed allowlist: {source_url}")
    request = Request(source_url, headers={"User-Agent": "SARA-independent-evidence-collector/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        final_url = response.geturl()
        final_domain = str(urlparse(final_url).hostname or "").lower()
        if urlparse(final_url).scheme != "https" or final_domain not in ALLOWED_DOMAINS:
            raise ValueError(f"redirect left the reviewed allowlist: {final_url}")
        payload = response.read(MAX_RESPONSE_BYTES + 1)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ValueError(f"source exceeds byte ceiling: {source_url}")
        content_type = str(response.headers.get("Content-Type", ""))
        charset = response.headers.get_content_charset() or "utf-8"
        content = _normalize_content(payload, content_type, charset)
        revision = (
            response.headers.get("ETag")
            or response.headers.get("Last-Modified")
            or entry.get("source_revision_hint")
            or ""
        )
    source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "schema": "sara-independent-source-row-v1",
        "record_id": f"external-{source_hash[:16]}",
        "content": content,
        "source_url": source_url,
        "source_ref": final_url,
        "source_domain": final_domain,
        "source_revision": str(revision),
        "license_hint": str(entry["license_hint"]),
        "task_type": "delayed",
        "source_hash": source_hash,
        "response_body_hash": hashlib.sha256(payload).hexdigest(),
        "collection_time": collection_time,
        "evidence_scope": "independent_external",
        "observed_only": True,
        "compliance_level": "allow",
        "near_duplicate_signature": hashlib.sha256(
            re.sub(r"\W+", " ", content.lower()).encode("utf-8")
        ).hexdigest()[:16],
        "content_origin": "fetched_authoritative_document",
        "catalog_stage": str(entry.get("catalog_stage", "")),
        "content_truncated": len(content) == MAX_CONTENT_CHARS,
    }


def merge_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    collected_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    refs: set[str] = set()
    hashes: set[str] = set()
    for row in (*existing_rows, *collected_rows):
        source_ref = str(row.get("source_ref") or row.get("source_url") or "")
        source_hash = str(row.get("source_hash") or row.get("material_hash") or "")
        if not source_ref or not source_hash:
            raise ValueError("every source row requires a source reference and hash")
        if source_ref in refs or source_hash in hashes:
            raise ValueError("duplicate source reference or content hash")
        refs.add(source_ref)
        hashes.add(source_hash)
        merged.append(dict(row))
    return merged


def build_manifest(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    indices: Dict[str, int] = defaultdict(int)
    manifest: List[Dict[str, Any]] = []
    for row in rows:
        domain = _domain(row)
        content = str(row.get("content", ""))
        material_hash = str(row.get("source_hash") or row.get("material_hash") or "")
        if domain not in ALLOWED_DOMAINS or not content or not material_hash:
            raise ValueError("source row is not eligible for the external manifest")
        signature = _signature(content)
        horizon_index = indices[domain]
        indices[domain] += 1
        manifest.append(
            {
                "schema": "sara-own-latent-manifest-row-v1",
                "manifest_id": f"architecture_migration_latent_{domain}_{horizon_index:06d}",
                "material_hash": material_hash,
                "source_url": str(row["source_url"]),
                "source_domain": domain,
                "source_ref": str(row.get("source_ref") or row["source_url"]),
                "source_type": "independent_external_documentation",
                "source_revision": str(row.get("source_revision", "")),
                "license_hint": str(row.get("license_hint", "")),
                "observed_only": True,
                "compliance_level": "allow",
                "quality_score": 1.0,
                "event_cost": len(signature),
                "language": "en",
                "material_type": "source_claim",
                "latent_cluster_id": f"external_{domain}_{horizon_index:06d}",
                "sparse_signature": signature,
                "evidence_scope": "independent_external",
                "collection_time": str(row.get("collection_time", "")),
                "migration_horizon_index": horizon_index,
            }
        )
    return manifest


def _atomic_write_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    resolved = ensure_parent_directory(path)
    parent = os.path.dirname(resolved)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=parent, delete=False, prefix=".collect-", suffix=".jsonl"
    ) as handle:
        temporary = handle.name
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, resolved)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-horizon", type=int, default=10)
    parser.add_argument("--raw-path", default=DEFAULT_RAW_PATH)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    collection_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        existing = _read_jsonl(args.raw_path)
        needed = select_needed_sources(existing, args.target_horizon)
        collected = []
        for entry in needed:
            try:
                collected.append(
                    fetch_source(entry, collection_time, args.timeout_seconds)
                )
            except OSError as exc:
                raise OSError(f"failed to fetch {entry['source_url']}: {exc}") from exc
        merged = merge_rows(existing, collected)
        manifest = build_manifest(merged)
        domain_horizons = defaultdict(list)
        for row in manifest:
            domain_horizons[str(row["source_domain"])].append(
                int(row["migration_horizon_index"])
            )
        if any(max(values, default=-1) < args.target_horizon for values in domain_horizons.values()):
            raise ValueError("collected manifest did not reach the requested horizon")
        _atomic_write_jsonl(args.raw_path, merged)
        _atomic_write_jsonl(args.manifest_path, manifest)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    report = {
        "schema": "sara-continual-horizon-external-collection-v1",
        "observed_only": True,
        "target_horizon": args.target_horizon,
        "existing_record_count": len(existing),
        "collected_record_count": len(collected),
        "fetched_authoritative_record_count": sum(
            str(row.get("content_origin", "")) == "fetched_authoritative_document"
            for row in merged
        ),
        "transcribed_source_record_count": sum(
            str(row.get("content_origin", "")) == "transcribed_source_excerpt"
            for row in merged
        ),
        "total_record_count": len(merged),
        "domain_horizons": dict(sorted(domain_horizons.items())),
        "source_domains": sorted(domain_horizons),
        "raw_path": os.path.realpath(args.raw_path),
        "manifest_path": os.path.realpath(args.manifest_path),
    }
    with open(ensure_parent_directory(args.report_path), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
