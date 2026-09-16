from sara_engine.evaluation.r2_bpi2012_hybrid_final import load_protocol


def test_final_protocol_fixes_router_and_single_attempt():
    protocol = load_protocol()
    assert protocol["selected_router"] == {"base_probability_cap": .75, "local_score_margin": .10,
        "minimum_base_support": 16, "probability_rule": "swap base and local predicted-class probabilities on override"}
    assert protocol["evaluation_partition"] == "frozen_test"
    assert protocol["execution_policy"]["final_attempts"] == 1
