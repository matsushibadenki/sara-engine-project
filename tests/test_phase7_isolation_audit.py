import importlib.util
import os
import sys


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "phase7_isolation_audit.py"))
    spec = importlib.util.spec_from_file_location("phase7_isolation_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(domain, timestamp, content, source_hash, revision, signature):
    return {"source_domain": domain, "collection_time": timestamp, "source_text": content, "source_hash": source_hash, "source_revision": revision, "near_duplicate_signature": signature}


def test_phase7_isolation_audit_accepts_independent_time_split():
    module = _load_module()
    report = module.build_report(
        [_row("train.example", "2026-01-01T00:00:00", "sparse local learning", "train-hash", "train-rev", "0000000000000000")],
        [_row("eval.example", "2026-02-01T00:00:00", "independent held out measure", "eval-hash", "eval-rev", "ffffffffffffffff")],
    )
    assert report["passed"] is True


def test_phase7_isolation_audit_rejects_hash_and_near_duplicate_overlap():
    module = _load_module()
    report = module.build_report(
        [_row("train.example", "2026-01-01T00:00:00", "shared material", "same-hash", "train-rev", "0000000000000000")],
        [_row("eval.example", "2026-02-01T00:00:00", "shared material", "same-hash", "eval-rev", "0000000000000001")],
    )
    assert report["passed"] is False
    assert report["checks"]["source_hash_isolated"] is False
    assert report["checks"]["near_duplicate_signature_isolated"] is False
