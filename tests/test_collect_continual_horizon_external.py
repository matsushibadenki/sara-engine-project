from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "data" / "collect_continual_horizon_external.py"
    spec = importlib.util.spec_from_file_location("collect_continual_horizon_external", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(domain: str, index: int) -> dict:
    content = f"Observed authoritative content for {domain} record {index}. " * 12
    return {
        "content": content,
        "source_url": f"https://{domain}/existing/{index}",
        "source_ref": f"https://{domain}/existing/{index}",
        "source_domain": domain,
        "source_hash": f"hash-{domain}-{index}",
        "source_revision": f"revision-{index}",
        "license_hint": "license",
        "collection_time": "2026-08-05T00:00:00Z",
    }


def test_select_needed_sources_reaches_horizon_without_reusing_existing_refs():
    module = _load_module()
    existing = [
        *(_row("docs.python.org", index) for index in range(3)),
        *(_row("www.rfc-editor.org", index) for index in range(3)),
    ]

    selected = module.select_needed_sources(existing, 10)

    assert len(selected) == 16
    assert {module._domain(row) for row in selected} == {
        "docs.python.org",
        "www.rfc-editor.org",
    }


def test_select_needed_sources_rejects_unreviewed_horizon_expansion():
    module = _load_module()
    existing = [
        *(_row("docs.python.org", index) for index in range(3)),
        *(_row("www.rfc-editor.org", index) for index in range(3)),
    ]

    with pytest.raises(ValueError, match="reviewed source catalog"):
        module.select_needed_sources(existing, 101)


def test_reviewed_catalog_reaches_horizon_30_with_unique_allowed_urls():
    module = _load_module()
    urls = [str(row["source_url"]) for row in module.SOURCE_CATALOG]
    domains = [module._domain(row) for row in module.SOURCE_CATALOG]

    assert len(urls) == len(set(urls)) == 196
    assert set(domains) == module.ALLOWED_DOMAINS
    assert all(url.startswith("https://") for url in urls)
    assert all(row["catalog_stage"] for row in module.SOURCE_CATALOG)

    existing = [
        *(_row("docs.python.org", index) for index in range(3)),
        *(_row("www.rfc-editor.org", index) for index in range(3)),
    ]
    selected = module.select_needed_sources(existing, 30)
    assert len(selected) == 56

    selected = module.select_needed_sources(existing, 100)
    assert len(selected) == 196

    stage_counts = {}
    for row in module.SOURCE_CATALOG:
        stage = str(row["catalog_stage"])
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    assert stage_counts == {
        "horizon_10_pilot": 16,
        "horizon_30_expansion": 40,
        "horizon_100_expansion": 140,
    }


def test_manifest_uses_contiguous_per_domain_horizons():
    module = _load_module()
    rows = [
        *(_row("docs.python.org", index) for index in range(4)),
        *(_row("www.rfc-editor.org", index) for index in range(5)),
    ]

    manifest = module.build_manifest(rows)

    assert [
        row["migration_horizon_index"]
        for row in manifest
        if row["source_domain"] == "docs.python.org"
    ] == [0, 1, 2, 3]
    assert [
        row["migration_horizon_index"]
        for row in manifest
        if row["source_domain"] == "www.rfc-editor.org"
    ] == [0, 1, 2, 3, 4]


def test_merge_rows_rejects_duplicate_content_hash():
    module = _load_module()
    first = _row("docs.python.org", 0)
    duplicate = _row("www.rfc-editor.org", 0)
    duplicate["source_hash"] = first["source_hash"]

    with pytest.raises(ValueError, match="duplicate"):
        module.merge_rows([first], [duplicate])
