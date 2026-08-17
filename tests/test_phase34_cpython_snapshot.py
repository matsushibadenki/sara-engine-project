from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.phase34_cpython_snapshot import (
    COMMIT,
    FILE_ALLOWLIST,
    MODULES,
    build_manifest,
    build_preregistration,
    derive_modules,
    fetch_source_entry,
    validate_preregistration,
)
from sara_engine.evaluation.phase34_cpython_git_snapshot import (
    build_manifest as build_git_manifest,
    build_preregistration as build_git_preregistration,
    build_rows as build_git_rows,
    validate_preregistration as validate_git_preregistration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "evaluation"
    / "phase34_memory_cache_factorial_independent_case_plan.json"
)


def _case_plan() -> dict:
    return json.loads(CASE_PLAN.read_text(encoding="utf-8"))


def _load_collector():
    path = PROJECT_ROOT / "scripts" / "data" / "collect_phase34_cpython_snapshot.py"
    spec = importlib.util.spec_from_file_location("collect_phase34_cpython_snapshot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, url: str, payload: bytes) -> None:
        self._url = url
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, amount: int) -> bytes:
        assert amount > len(self._payload)
        return self._payload


def test_case_plan_derives_exact_frozen_thirty_file_allowlist():
    assert derive_modules(_case_plan()) == list(MODULES)
    assert len(MODULES) == len(FILE_ALLOWLIST) == 30
    assert len(set(FILE_ALLOWLIST)) == 30


def test_preregistration_is_commit_pinned_and_tamper_evident():
    registration = build_preregistration(_case_plan())

    validation = validate_preregistration(registration)

    assert validation["valid"] is True
    assert registration["commit"] == COMMIT
    assert registration["source_count"] == 30
    assert all(COMMIT in source["source_url"] for source in registration["sources"])

    registration["sources"][0]["repository_path"] = "Doc/library/changed.rst"
    assert validate_preregistration(registration)["valid"] is False


def test_preregistration_rejects_a_case_plan_with_a_changed_source_set():
    case_plan = _case_plan()
    case_plan["cases"][0]["stream_source_refs"].append(
        "https://docs.python.org/3.14/library/zipfile.html"
    )

    with pytest.raises(ValueError, match="frozen 30-module"):
        build_preregistration(case_plan)


def test_fetch_source_entry_rejects_redirect_outside_commit_path():
    registration = build_preregistration(_case_plan())
    entry = registration["sources"][0]
    payload = ("Authoritative CPython source documentation.\n" * 10).encode("utf-8")

    def bad_opener(request, timeout):
        del request, timeout
        return _Response("https://example.com/changed.rst", payload)

    with pytest.raises(ValueError, match="commit-addressed"):
        fetch_source_entry(entry, opener=bad_opener)


def test_collection_and_manifest_preserve_allowlist_order_and_prior_time():
    collector = _load_collector()
    registration = build_preregistration(_case_plan())

    def fetcher(entry, *, timeout_seconds, collection_time):
        del timeout_seconds
        content = (f"{entry['module']} CPython documentation source.\n" * 12)
        import hashlib

        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return {
            "schema": "sara-phase34-cpython-source-row-v1",
            "record_id": f"phase34-cpython-{entry['module']}-{digest[:16]}",
            "module": entry["module"],
            "repository": "python/cpython",
            "repository_path": entry["repository_path"],
            "content": content,
            "source_url": entry["source_url"],
            "source_ref": entry["source_url"],
            "source_domain": "raw.githubusercontent.com",
            "source_revision": COMMIT,
            "source_tag": "v3.14.6",
            "license_hint": "Python Software Foundation License Version 2",
            "source_hash": digest,
            "response_body_hash": digest,
            "collection_time": collection_time,
            "evidence_scope": "independent_external_commit_snapshot",
            "observed_only": True,
            "compliance_level": "allow",
            "content_origin": "fetched_commit_addressed_source",
            "predecessor_source_ref": entry["predecessor_source_ref"],
            "content_truncated": False,
        }

    rows = collector.collect(
        registration,
        timeout_seconds=1.0,
        fetcher=fetcher,
    )
    manifest = build_manifest(rows, registration)

    assert [row["repository_path"] for row in rows] == list(FILE_ALLOWLIST)
    assert [row["snapshot_index"] for row in manifest] == list(range(30))
    assert all(row["protocol_fingerprint"] == registration["protocol_fingerprint"] for row in manifest)

    replayed = collector.collect(
        registration,
        timeout_seconds=1.0,
        existing_rows=rows,
        fetcher=fetcher,
    )
    assert replayed == rows


def test_git_fallback_is_a_separate_commit_pinned_registration():
    registration = build_git_preregistration(_case_plan())

    assert validate_git_preregistration(registration)["valid"] is True
    assert registration["experiment_id"] == "phase34-cpython-v3.14.6-git-source-snapshot-v1"
    assert registration["collection_policy"]["transport"] == "git_smart_http_shallow_fetch"
    assert registration["claim_boundaries"]["raw_http_failure_is_not_erased"] is True


def test_git_fallback_builds_thirty_hash_bound_rows_and_manifest():
    registration = build_git_preregistration(_case_plan())

    def load_blob(path: str) -> bytes:
        return (f"CPython commit blob for {path}.\n" * 12).encode("utf-8")

    rows = build_git_rows(
        registration,
        load_blob,
        collection_time="2026-08-18T00:00:00Z",
    )
    manifest = build_git_manifest(rows, registration)

    assert len(rows) == len(manifest) == 30
    assert [row["repository_path"] for row in rows] == list(FILE_ALLOWLIST)
    assert all(row["source_revision"] == COMMIT for row in rows)
    assert all(row["acquisition_transport"] == "git_smart_http_shallow_fetch" for row in rows)
