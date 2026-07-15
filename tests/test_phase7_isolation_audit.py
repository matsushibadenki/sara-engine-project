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
    return {"source_domain": domain, "collection_time": timestamp, "source_text": content, "source_hash": source_hash, "source_revision": revision, "near_duplicate_signature": signature, "evidence_scope": "independent_external"}


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


def test_phase7_isolation_audit_normalizes_timezone_and_numeric_times():
    module = _load_module()
    report = module.build_report(
        [_row("train.example", "1767225600", "train", "train-hash", "train-rev", "0000000000000000")],
        [_row("eval.example", "2026-02-01T00:00:00+09:00", "eval", "eval-hash", "eval-rev", "ffffffffffffffff")],
    )
    assert report["checks"]["time_split_isolated"] is True


def test_phase7_isolation_audit_rejects_unparseable_times():
    module = _load_module()
    report = module.build_report(
        [_row("train.example", "not-a-time", "train", "train-hash", "train-rev", "0000000000000000")],
        [_row("eval.example", "2026-02-01T00:00:00Z", "eval", "eval-hash", "eval-rev", "ffffffffffffffff")],
    )
    assert report["checks"]["time_split_isolated"] is False


def test_phase7_isolation_audit_rejects_non_finite_numeric_times():
    module = _load_module()
    report = module.build_report(
        [_row("train.example", "nan", "train", "train-hash", "train-rev", "0000000000000000")],
        [_row("eval.example", "inf", "eval", "eval-hash", "eval-rev", "ffffffffffffffff")],
    )
    assert report["checks"]["time_split_isolated"] is False


def test_phase7_isolation_audit_rejects_negative_signature_distance():
    module = _load_module()
    try:
        module.build_report([], [], max_signature_hamming_distance=-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative signature distance should be rejected")


def test_phase7_isolation_audit_rejects_fractional_signature_distance():
    module = _load_module()
    try:
        module.build_report([], [], max_signature_hamming_distance=1.5)
    except ValueError as exc:
        assert "non-negative integer" in str(exc)
    else:
        raise AssertionError("fractional signature distance should be rejected")


def test_phase7_isolation_audit_rejects_unconfirmed_evidence_scope():
    module = _load_module()
    train = _row("train.example", "2026-01-01T00:00:00Z", "train", "train-hash", "train-rev", "0000000000000000")
    train.pop("evidence_scope")
    report = module.build_report(
        [train],
        [_row("eval.example", "2026-02-01T00:00:00Z", "eval", "eval-hash", "eval-rev", "ffffffffffffffff")],
    )
    assert report["passed"] is False
    assert report["checks"]["independent_evidence_scope_valid"] is False


def test_phase7_isolation_audit_rejects_malformed_duplicate_signature():
    module = _load_module()
    report = module.build_report(
        [_row("train.example", "2026-01-01T00:00:00Z", "train", "train-hash", "train-rev", "not-hex")],
        [_row("eval.example", "2026-02-01T00:00:00Z", "eval", "eval-hash", "eval-rev", "ffffffffffffffff")],
    )
    assert report["passed"] is False
    assert report["checks"]["near_duplicate_signature_format_valid"] is False
