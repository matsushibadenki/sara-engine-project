"""Immutable contract and bounded collector helpers for the Phase 34 CPython snapshot."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SCHEMA = "sara-phase34-cpython-source-snapshot-preregistration-v1"
EXPERIMENT_ID = "phase34-cpython-v3.14.6-source-snapshot-v1"
REPOSITORY = "python/cpython"
TAG = "v3.14.6"
TAG_OBJECT = "8594736f5057fdc979d42d2135895d56274589a8"
COMMIT = "c63aec69bd59c55314c06c23f4c22c03de76fe45"
ALLOWED_HOST = "raw.githubusercontent.com"
MAX_FILE_BYTES = 2_000_000
MODULES = (
    "asyncio",
    "builtins",
    "colorsys",
    "contextlib",
    "csv",
    "dataclasses",
    "enum",
    "fractions",
    "functools",
    "gettext",
    "hashlib",
    "http",
    "itertools",
    "json",
    "keyword",
    "logging",
    "mimetypes",
    "os",
    "pkgutil",
    "pty",
    "random",
    "sched",
    "smtplib",
    "sqlite3",
    "time",
    "tomllib",
    "typing",
    "urllib.request",
    "uuid",
    "zoneinfo",
)
FILE_ALLOWLIST = tuple(f"Doc/library/{module}.rst" for module in MODULES)


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_modules(case_plan: Mapping[str, Any]) -> List[str]:
    modules = set()
    for case in case_plan.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        for source_ref in case.get("stream_source_refs", []):
            match = re.fullmatch(
                r"https://docs\.python\.org/3\.14/library/([a-z0-9_.]+)\.html",
                str(source_ref),
            )
            if match:
                modules.add(match.group(1))
    return sorted(modules)


def _source_entry(module: str) -> Dict[str, str]:
    path = f"Doc/library/{module}.rst"
    return {
        "module": module,
        "repository_path": path,
        "source_url": (
            f"https://{ALLOWED_HOST}/{REPOSITORY}/{COMMIT}/{path}"
        ),
        "predecessor_source_ref": (
            f"https://docs.python.org/3.14/library/{module}.html"
        ),
    }


def build_preregistration(case_plan: Mapping[str, Any]) -> Dict[str, Any]:
    derived = derive_modules(case_plan)
    if derived != list(MODULES):
        raise ValueError("case plan does not resolve to the frozen 30-module allowlist")
    candidate: Dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "registered_before_collection": True,
        "repository": REPOSITORY,
        "tag": TAG,
        "tag_object": TAG_OBJECT,
        "commit": COMMIT,
        "allowed_host": ALLOWED_HOST,
        "file_allowlist": list(FILE_ALLOWLIST),
        "sources": [_source_entry(module) for module in MODULES],
        "source_count": len(MODULES),
        "case_plan_fingerprint": canonical_digest(dict(case_plan)),
        "collection_policy": {
            "https_only": True,
            "commit_addressed_paths_only": True,
            "redirect_must_preserve_host_commit_and_path": True,
            "max_file_bytes": MAX_FILE_BYTES,
            "utf8_required": True,
            "content_truncation_allowed": False,
            "replace_executed_v2_sources": False,
            "overwrite_on_mismatch": False,
        },
        "claim_boundaries": {
            "immutable_source_identity_only": True,
            "semantic_delayed_recall_allowed": False,
            "language_understanding_claim_allowed": False,
            "promotion_ready": False,
        },
    }
    candidate["protocol_fingerprint"] = canonical_digest(candidate)
    return candidate


def validate_preregistration(value: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    fingerprint_input = dict(value)
    declared = fingerprint_input.pop("protocol_fingerprint", None)
    computed = canonical_digest(fingerprint_input)
    exact = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "registered_before_collection": True,
        "repository": REPOSITORY,
        "tag": TAG,
        "tag_object": TAG_OBJECT,
        "commit": COMMIT,
        "allowed_host": ALLOWED_HOST,
        "file_allowlist": list(FILE_ALLOWLIST),
        "sources": [_source_entry(module) for module in MODULES],
        "source_count": len(MODULES),
        "collection_policy": {
            "https_only": True,
            "commit_addressed_paths_only": True,
            "redirect_must_preserve_host_commit_and_path": True,
            "max_file_bytes": MAX_FILE_BYTES,
            "utf8_required": True,
            "content_truncation_allowed": False,
            "replace_executed_v2_sources": False,
            "overwrite_on_mismatch": False,
        },
        "claim_boundaries": {
            "immutable_source_identity_only": True,
            "semantic_delayed_recall_allowed": False,
            "language_understanding_claim_allowed": False,
            "promotion_ready": False,
        },
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            errors.append(f"frozen_snapshot_mismatch:{key}")
    case_plan_fingerprint = value.get("case_plan_fingerprint")
    if not (
        isinstance(case_plan_fingerprint, str)
        and len(case_plan_fingerprint) == 64
        and all(character in "0123456789abcdef" for character in case_plan_fingerprint)
    ):
        errors.append("invalid_case_plan_fingerprint")
    if declared != computed:
        errors.append("protocol_fingerprint_mismatch")
    return {
        "valid": not errors,
        "errors": errors,
        "declared_fingerprint": declared,
        "computed_fingerprint": computed,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_final_url(final_url: str, expected_path: str) -> None:
    parsed = urlparse(final_url)
    expected_url_path = f"/{REPOSITORY}/{COMMIT}/{expected_path}"
    if (
        parsed.scheme != "https"
        or parsed.hostname != ALLOWED_HOST
        or parsed.path != expected_url_path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("retrieval left the preregistered commit-addressed source path")


def fetch_source_entry(
    entry: Mapping[str, Any],
    *,
    timeout_seconds: float = 20.0,
    collection_time: Optional[str] = None,
    opener: Callable[..., Any] = urlopen,
) -> Dict[str, Any]:
    path = str(entry.get("repository_path", ""))
    source_url = str(entry.get("source_url", ""))
    if path not in FILE_ALLOWLIST or entry != _source_entry(str(entry.get("module", ""))):
        raise ValueError("source entry is outside the preregistered allowlist")
    _validate_final_url(source_url, path)
    request = Request(
        source_url,
        headers={"User-Agent": "SARA-commit-snapshot-collector/1.0"},
    )
    with opener(request, timeout=timeout_seconds) as response:
        final_url = str(response.geturl())
        _validate_final_url(final_url, path)
        payload = response.read(MAX_FILE_BYTES + 1)
    if len(payload) > MAX_FILE_BYTES:
        raise ValueError("source file exceeds the preregistered byte ceiling")
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("source file is not valid UTF-8") from exc
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    if len(content.strip()) < 200:
        raise ValueError("source file is too short to qualify")
    source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "schema": "sara-phase34-cpython-source-row-v1",
        "record_id": f"phase34-cpython-{entry['module']}-{source_hash[:16]}",
        "module": str(entry["module"]),
        "repository": REPOSITORY,
        "repository_path": path,
        "content": content,
        "source_url": source_url,
        "source_ref": final_url,
        "source_domain": ALLOWED_HOST,
        "source_revision": COMMIT,
        "source_tag": TAG,
        "license_hint": "Python Software Foundation License Version 2",
        "source_hash": source_hash,
        "response_body_hash": hashlib.sha256(payload).hexdigest(),
        "collection_time": collection_time or utc_now(),
        "evidence_scope": "independent_external_commit_snapshot",
        "observed_only": True,
        "compliance_level": "allow",
        "content_origin": "fetched_commit_addressed_source",
        "predecessor_source_ref": str(entry["predecessor_source_ref"]),
        "content_truncated": False,
    }


def sparse_signature(content: str, width: int = 4096) -> List[int]:
    tokens = re.findall(r"[a-z0-9]+", content.lower())
    return sorted(
        {
            int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % width
            for token in tokens
        }
    )


def build_manifest(
    rows: Sequence[Mapping[str, Any]], registration: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    validation = validate_preregistration(registration)
    if not validation["valid"]:
        raise ValueError("invalid CPython snapshot preregistration")
    by_path = {str(row.get("repository_path", "")): row for row in rows}
    if set(by_path) != set(FILE_ALLOWLIST) or len(rows) != len(FILE_ALLOWLIST):
        raise ValueError("collected rows do not match the frozen file allowlist")
    hashes = [str(row.get("source_hash", "")) for row in rows]
    if not all(hashes) or len(set(hashes)) != len(hashes):
        raise ValueError("snapshot source hashes must be present and unique")
    manifest: List[Dict[str, Any]] = []
    for index, path in enumerate(FILE_ALLOWLIST):
        row = by_path[path]
        content = str(row.get("content", ""))
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != row.get("source_hash"):
            raise ValueError("snapshot content hash mismatch")
        signature = sparse_signature(content)
        manifest.append(
            {
                "schema": "sara-phase34-cpython-source-manifest-row-v1",
                "manifest_id": f"phase34_cpython_snapshot_{index:03d}",
                "snapshot_index": index,
                "module": str(row["module"]),
                "repository_path": path,
                "material_hash": str(row["source_hash"]),
                "source_ref": str(row["source_ref"]),
                "source_domain": ALLOWED_HOST,
                "source_revision": COMMIT,
                "source_tag": TAG,
                "predecessor_source_ref": str(row["predecessor_source_ref"]),
                "license_hint": str(row["license_hint"]),
                "observed_only": True,
                "compliance_level": "allow",
                "quality_score": 1.0,
                "event_cost": len(signature),
                "language": "en",
                "material_type": "commit_addressed_source_claim",
                "sparse_signature": signature,
                "evidence_scope": "independent_external_commit_snapshot",
                "collection_time": str(row["collection_time"]),
                "protocol_fingerprint": str(registration["protocol_fingerprint"]),
            }
        )
    return manifest


__all__ = [
    "ALLOWED_HOST",
    "COMMIT",
    "EXPERIMENT_ID",
    "FILE_ALLOWLIST",
    "MAX_FILE_BYTES",
    "MODULES",
    "REPOSITORY",
    "SCHEMA",
    "TAG",
    "TAG_OBJECT",
    "build_manifest",
    "build_preregistration",
    "canonical_digest",
    "derive_modules",
    "fetch_source_entry",
    "sparse_signature",
    "utc_now",
    "validate_preregistration",
]
