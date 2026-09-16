from scripts.data.audit_r2_sepsis import build_audit

def test_sepsis_audit_preserves_irregular_real_event_contract():
    audit = build_audit()
    assert audit["source"]["md5"] == "b5671166ac71eb20680d3c74616c43d2"
    assert audit["trace_count"] >= 1000 and audit["event_count"] >= 15000
    assert audit["activity_count"] == 16
    assert audit["inter_event_seconds"]["negative_count"] == 0
    assert audit["inter_event_seconds"]["distinct_values"] > 100
