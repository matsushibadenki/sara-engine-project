from sara_engine.evaluation.r2_sepsis_normalized import load_protocol
def test_normalized_protocol_is_hash_bound_and_keeps_evaluation_sealed():
    protocol=load_protocol()
    assert protocol["router"]["target_override_fraction_of_all_predictions"]==.10
    assert protocol["execution_policy"]["development_metrics_forbidden"]
    assert protocol["execution_policy"]["frozen_test_metrics_forbidden"]
