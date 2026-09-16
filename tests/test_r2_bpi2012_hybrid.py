from sara_engine.evaluation.r2_bpi2012_hybrid import load_protocol


def test_hybrid_protocol_is_hash_bound_and_forbids_exposed_splits():
    protocol = load_protocol()
    assert protocol["router"]["candidate_count"] == 48
    assert protocol["execution_policy"]["existing_development_metrics_forbidden"]
    assert protocol["execution_policy"]["frozen_test_metrics_forbidden"]
    assert protocol["execution_policy"]["attempts"] == 1
