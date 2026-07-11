import importlib.util
import os
import sys


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "architecture_migration_external_gate.py"))
    spec = importlib.util.spec_from_file_location("architecture_migration_external_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(domain: str, identifier: str):
    return {
        "schema": "sara-own-latent-manifest-row-v1", "observed_only": True,
        "compliance_level": "allow", "source_url": f"https://{domain}/record/{identifier}",
        "material_hash": f"hash-{identifier}", "manifest_id": f"manifest-{identifier}",
        "latent_cluster_id": f"latent-{identifier}", "sparse_signature": [1, 3, 5, ord(identifier)],
        "quality_score": 0.9, "event_cost": 6,
    }


def test_external_gate_requires_independent_provenance_and_preserves_replay():
    module = _load_module()
    report = module.build_report(
        tuple(
            _row(domain, identifier)
            for domain, identifiers in (
                ("docs.alpha.test", ("a", "b", "c")),
                ("docs.beta.test", ("d", "e", "f")),
            )
            for identifier in identifiers
        )
    )
    assert report["passed"] is True
    assert report["external_provenance_qualified"] is True
    assert report["metrics"]["target_replay_recall"] == 1.0

    blocked = module.build_report((_row("docs.gamma.test", "x"),))
    assert blocked["passed"] is False
    assert "insufficient_independent_source_sites" in blocked["blocked_reasons"]
    assert "insufficient_long_horizon_records_per_site" in blocked["blocked_reasons"]
