from scripts.data.audit_r2_bpi2012 import build_audit


def test_hash_pinned_bpi2012_source_is_irregular_and_chronological():
    audit = build_audit()
    assert audit["trace_count"] == 13087
    assert audit["event_count"] == 262200
    assert audit["activity_count"] == 24
    assert audit["inter_event_seconds"]["distinct_values"] == 148911
    assert audit["inter_event_seconds"]["negative_count"] == 0
    assert audit["inter_event_seconds"]["p90"] > audit["inter_event_seconds"]["median"] * 100


def test_case_split_boundaries_are_ordered():
    audit = build_audit()
    assert audit["time_start"] < audit["case_start_percentiles"]["p70"]
    assert audit["case_start_percentiles"]["p70"] < audit["case_start_percentiles"]["p85"]
    assert audit["case_start_percentiles"]["p85"] < audit["time_end"]
