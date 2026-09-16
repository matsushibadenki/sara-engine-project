from sara_engine.evaluation.r2_sepsis_normalized_final import load_protocol
def test_final_protocol_fixes_threshold_and_one_attempt():
    protocol=load_protocol()
    assert protocol["fixed_normalized_score_threshold"]==0.6223091976516634
    assert protocol["evaluation_partition"]=="frozen_test"
    assert protocol["execution_policy"]["final_attempts"]==1
