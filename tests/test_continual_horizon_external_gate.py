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
    assert report["promotion_allowed"] is False
    assert report["metrics"]["eligible_record_count"] == 6
    assert report["metrics"]["source_domain_count"] == 2
    assert report["promotion_checks"]["required_horizon_buckets_present"] is False
    assert report["next_actions"]


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
