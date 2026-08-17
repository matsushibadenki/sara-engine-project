from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "continual_horizon_external_gate.py"
    spec = importlib.util.spec_from_file_location("continual_horizon_external_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_continual_horizon_external_gate_accepts_existing_independent_manifest():
    module = _load_module()
    manifest = PROJECT_ROOT / "data" / "processed" / "autobot" / "architecture_migration_latent_manifest.jsonl"
    report = module.build_report(module._load(str(manifest)))

    assert report["passed"] is True
    assert report["promotion_allowed"] is True
    assert report["metrics"]["eligible_record_count"] == 202
    assert report["metrics"]["source_domain_count"] == 2
    assert report["metrics"]["min_records_per_domain"] == 101
    assert report["metrics"]["minimum_domain_horizon_span"] == 100
    assert report["horizon_bucket_coverage"]["10"] is True
    assert report["horizon_bucket_coverage"]["30"] is True
    assert report["horizon_bucket_coverage"]["100"] is True
    assert report["promotion_checks"]["required_horizon_buckets_present"] is True
    assert report["next_actions"] == []


def test_continual_horizon_external_gate_rejects_duplicate_hash():
    module = _load_module()
    rows = [
        {
            "evidence_scope": "independent_external",
            "source_domain": "one.example",
            "material_hash": "same",
            "source_ref": "ref-1",
            "source_revision": "rev-1",
            "collection_time": "2026-07-19",
            "migration_horizon_index": 0,
            "observed_only": True,
            "compliance_level": "allow",
        },
        {
            "evidence_scope": "independent_external",
            "source_domain": "two.example",
            "material_hash": "same",
            "source_ref": "ref-2",
            "source_revision": "rev-2",
            "collection_time": "2026-07-19",
            "migration_horizon_index": 0,
            "observed_only": True,
            "compliance_level": "allow",
        },
    ]
    report = module.build_report(rows)

    assert report["passed"] is False
    assert report["checks"]["unique_material_hashes"] is False


def test_continual_horizon_external_gate_rejects_sparse_or_global_domain_indices():
    module = _load_module()
    rows = []
    for domain, indices in (("one.example", (0, 1, 2)), ("two.example", (3, 4, 5))):
        for index in indices:
            rows.append(
                {
                    "evidence_scope": "independent_external",
                    "source_domain": domain,
                    "material_hash": f"hash-{domain}-{index}",
                    "source_ref": f"https://{domain}/{index}",
                    "source_revision": f"rev-{index}",
                    "collection_time": "2026-07-19",
                    "migration_horizon_index": index,
                    "observed_only": True,
                    "compliance_level": "allow",
                }
            )

    report = module.build_report(rows)

    assert report["passed"] is False
    assert report["checks"]["contiguous_horizons_per_domain"] is False
