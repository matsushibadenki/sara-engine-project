import importlib.util
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


def test_evidence_cycle_qualifies_and_gates_external_records(tmp_path):
    module = _load("architecture_migration_evidence_cycle.py")
    rows = [
        {"schema": "sara-own-latent-manifest-row-v1", "source_url": f"https://{domain}/r/{index}", "material_hash": f"hash-{domain}-{index}", "manifest_id": f"id-{domain}-{index}", "latent_cluster_id": "latent", "sparse_signature": [index + 1, 11, 13], "observed_only": True, "compliance_level": "allow", "quality_score": 0.9}
        for domain in ("alpha.test", "beta.test") for index in range(3)
    ]
    path = tmp_path / "latent.jsonl"
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = module.run_cycle(str(path))
    assert report["promotion_eligible"] is True
    assert report["qualified_record_count"] == 6
    assert report["collection_targets"]["target_count"] == 0
