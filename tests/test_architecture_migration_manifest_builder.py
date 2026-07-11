import importlib.util
import json
import os
import sys


def _load(name: str):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", name))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_builder_normalizes_external_provenance_for_gate():
    builder = _load("architecture_migration_manifest_builder.py")
    gate = _load("architecture_migration_external_gate.py")
    rows = [
        {"schema": "sara-own-latent-manifest-row-v1", "source_url": f"https://{domain}/r/{index}", "material_hash": f"hash-{domain}-{index}", "manifest_id": f"id-{domain}-{index}", "latent_cluster_id": "latent", "sparse_signature": [index + 1, 11, 13], "observed_only": True, "compliance_level": "allow", "quality_score": 0.9}
        for domain in ("alpha.test", "beta.test") for index in range(3)
    ]
    manifest = builder.build_manifest(rows)
    assert len(manifest) == 6
    assert all(row["schema"] == "sara-architecture-migration-source-row-v1" for row in manifest)
    assert {row["source_site"] for row in manifest} == {"alpha.test", "beta.test"}
    assert all(len(row["provenance_digest"]) == 64 for row in manifest)
    assert gate.build_report(manifest)["passed"] is True
