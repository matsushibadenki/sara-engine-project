"""Git-transport fallback contract for the immutable Phase 34 CPython snapshot."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Mapping, Sequence

from sara_engine.evaluation.phase34_cpython_snapshot import (
    COMMIT,
    FILE_ALLOWLIST,
    MAX_FILE_BYTES,
    MODULES,
    REPOSITORY,
    TAG,
    TAG_OBJECT,
    canonical_digest,
    derive_modules,
    sparse_signature,
)


SCHEMA = "sara-phase34-cpython-git-source-snapshot-preregistration-v1"
EXPERIMENT_ID = "phase34-cpython-v3.14.6-git-source-snapshot-v1"
REPOSITORY_URL = "https://github.com/python/cpython.git"


def _source_entry(module: str) -> Dict[str, str]:
    path = f"Doc/library/{module}.rst"
    return {
        "module": module,
        "repository_path": path,
        "source_ref": f"https://github.com/{REPOSITORY}/blob/{COMMIT}/{path}",
        "predecessor_source_ref": f"https://docs.python.org/3.14/library/{module}.html",
    }


def build_preregistration(case_plan: Mapping[str, Any]) -> Dict[str, Any]:
    if derive_modules(case_plan) != list(MODULES):
        raise ValueError("case plan does not resolve to the frozen 30-module allowlist")
    candidate: Dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "registered_before_collection": True,
        "repository": REPOSITORY,
        "repository_url": REPOSITORY_URL,
        "tag": TAG,
        "tag_object": TAG_OBJECT,
        "commit": COMMIT,
        "file_allowlist": list(FILE_ALLOWLIST),
        "sources": [_source_entry(module) for module in MODULES],
        "source_count": len(MODULES),
        "case_plan_fingerprint": canonical_digest(dict(case_plan)),
        "collection_policy": {
            "transport": "git_smart_http_shallow_fetch",
            "fetch_ref": COMMIT,
            "depth": 1,
            "tags": False,
            "submodules": False,
            "lfs": False,
            "head_must_equal_commit": True,
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
            "raw_http_failure_is_not_erased": True,
        },
    }
    candidate["protocol_fingerprint"] = canonical_digest(candidate)
    return candidate


def validate_preregistration(value: Mapping[str, Any]) -> Dict[str, Any]:
    fingerprint_input = dict(value)
    declared = fingerprint_input.pop("protocol_fingerprint", None)
    computed = canonical_digest(fingerprint_input)
    errors: List[str] = []
    case_plan_fingerprint = value.get("case_plan_fingerprint")
    if not (
        isinstance(case_plan_fingerprint, str)
        and len(case_plan_fingerprint) == 64
        and all(character in "0123456789abcdef" for character in case_plan_fingerprint)
    ):
        errors.append("invalid_case_plan_fingerprint")
    expected = build_preregistration(
        {
            "cases": [
                {
                    "stream_source_refs": [
                        f"https://docs.python.org/3.14/library/{module}.html"
                        for module in MODULES
                    ]
                }
            ]
        }
    )
    for key in (
        "schema",
        "experiment_id",
        "registered_before_collection",
        "repository",
        "repository_url",
        "tag",
        "tag_object",
        "commit",
        "file_allowlist",
        "sources",
        "source_count",
        "collection_policy",
        "claim_boundaries",
    ):
        if value.get(key) != expected.get(key):
            errors.append(f"frozen_git_snapshot_mismatch:{key}")
    if declared != computed:
        errors.append("protocol_fingerprint_mismatch")
    return {"valid": not errors, "errors": errors, "computed_fingerprint": computed}


def build_rows(
    registration: Mapping[str, Any],
    blob_loader: Callable[[str], bytes],
    *,
    collection_time: str,
) -> List[Dict[str, Any]]:
    validation = validate_preregistration(registration)
    if not validation["valid"]:
        raise ValueError("invalid Git snapshot preregistration: " + "; ".join(validation["errors"]))
    rows: List[Dict[str, Any]] = []
    for entry in registration["sources"]:
        path = str(entry["repository_path"])
        payload = blob_loader(path)
        if len(payload) > MAX_FILE_BYTES:
            raise ValueError(f"source file exceeds byte ceiling: {path}")
        try:
            content = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError as exc:
            raise ValueError(f"source file is not valid UTF-8: {path}") from exc
        if len(content.strip()) < 200:
            raise ValueError(f"source file is too short to qualify: {path}")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        rows.append(
            {
                "schema": "sara-phase34-cpython-git-source-row-v1",
                "record_id": f"phase34-cpython-git-{entry['module']}-{digest[:16]}",
                "module": str(entry["module"]),
                "repository": REPOSITORY,
                "repository_url": REPOSITORY_URL,
                "repository_path": path,
                "content": content,
                "source_url": str(entry["source_ref"]),
                "source_ref": str(entry["source_ref"]),
                "source_domain": "github.com",
                "source_revision": COMMIT,
                "source_tag": TAG,
                "license_hint": "Python Software Foundation License Version 2",
                "source_hash": digest,
                "response_body_hash": hashlib.sha256(payload).hexdigest(),
                "collection_time": collection_time,
                "evidence_scope": "independent_external_commit_snapshot",
                "observed_only": True,
                "compliance_level": "allow",
                "content_origin": "fetched_git_commit_blob",
                "acquisition_transport": "git_smart_http_shallow_fetch",
                "predecessor_source_ref": str(entry["predecessor_source_ref"]),
                "content_truncated": False,
            }
        )
    return rows


def build_manifest(rows: Sequence[Mapping[str, Any]], registration: Mapping[str, Any]) -> List[Dict[str, Any]]:
    by_path = {str(row.get("repository_path", "")): row for row in rows}
    if set(by_path) != set(FILE_ALLOWLIST) or len(rows) != len(FILE_ALLOWLIST):
        raise ValueError("collected rows do not match the frozen file allowlist")
    manifest: List[Dict[str, Any]] = []
    for index, path in enumerate(FILE_ALLOWLIST):
        row = by_path[path]
        content = str(row["content"])
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != row["source_hash"]:
            raise ValueError("snapshot content hash mismatch")
        signature = sparse_signature(content)
        manifest.append(
            {
                "schema": "sara-phase34-cpython-git-source-manifest-row-v1",
                "manifest_id": f"phase34_cpython_git_snapshot_{index:03d}",
                "snapshot_index": index,
                "module": row["module"],
                "repository_path": path,
                "material_hash": row["source_hash"],
                "source_ref": row["source_ref"],
                "source_domain": "github.com",
                "source_revision": COMMIT,
                "source_tag": TAG,
                "predecessor_source_ref": row["predecessor_source_ref"],
                "license_hint": row["license_hint"],
                "observed_only": True,
                "compliance_level": "allow",
                "quality_score": 1.0,
                "event_cost": len(signature),
                "language": "en",
                "material_type": "commit_addressed_source_claim",
                "sparse_signature": signature,
                "evidence_scope": "independent_external_commit_snapshot",
                "collection_time": row["collection_time"],
                "protocol_fingerprint": registration["protocol_fingerprint"],
            }
        )
    return manifest
