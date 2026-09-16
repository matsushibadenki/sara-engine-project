from sara_engine.evaluation.r2_bpi2012_development import decide, load_protocol


def test_decision_blocks_test_when_scalar_trace_differs():
    protocol=load_protocol()
    metric=lambda acc,f1,rare=.2:{"top1_accuracy":acc,"macro_f1":f1,"rare_activity_mean_recall":rare}
    result=lambda acc,f1,digest="same":{"development":metric(acc,f1),"prediction_trace_sha256":digest,"resources":{"contracts_passed":True}}
    arms={"snn_sparse_mistake":result(.83,.59),"scalar_sparse_mistake":result(.83,.59),
          "snn_constant_gap":result(.81,.57),"snn_frozen":result(.20,.02),"snn_shuffled_outcomes":result(.30,.10),
          "online_second_order_transition":result(.80,.55)}
    assert decide(protocol,arms)["frozen_test_authorized"]
    arms["scalar_sparse_mistake"]["prediction_trace_sha256"]="different"
    assert not decide(protocol,arms)["frozen_test_authorized"]
