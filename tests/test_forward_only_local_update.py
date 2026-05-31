from sara_engine.nn.forward_only_local_update import (
    ForwardOnlyLocalUpdateTrace,
    evaluate_forward_only_local_update_trace,
)


def test_forward_only_local_update_strengthens_causal_pair_without_bptt() -> None:
    trace = ForwardOnlyLocalUpdateTrace(capacity=4, learning_rate=0.25)

    update = trace.update(pre_events=[1], post_events=[10], credit=1.0)

    assert update["bptt_used"] is False
    assert update["weight_updates"] == 1
    assert trace.read_weight(1, 10) == 0.25
    assert update["state_budget_ok"] is True


def test_forward_only_local_update_keeps_weights_bounded() -> None:
    trace = ForwardOnlyLocalUpdateTrace(capacity=4, learning_rate=0.5, max_abs_weight=1.0)

    for _ in range(8):
        trace.update(pre_events=[1], post_events=[10], credit=1.0)
    positive_weight = trace.read_weight(1, 10)
    for _ in range(8):
        trace.update(pre_events=[1], post_events=[10], credit=-1.0)
    negative_weight = trace.read_weight(1, 10)

    assert positive_weight == 1.0
    assert negative_weight == -1.0
    assert trace.snapshot()["state_budget_ok"] is True


def test_forward_only_local_update_enforces_sparse_state_budget() -> None:
    trace = ForwardOnlyLocalUpdateTrace(capacity=3, learning_rate=0.25)

    update = trace.update(pre_events=[1, 2, 3], post_events=[10, 20], credit=0.5)
    snapshot = trace.snapshot()

    assert update["state_budget_ok"] is True
    assert update["evicted_count"] > 0
    assert len(snapshot["entries"]) <= 3


def test_forward_only_local_update_evaluation_reports_observed_metrics() -> None:
    report = evaluate_forward_only_local_update_trace()

    assert report["observed_only"] is True
    assert report["metrics"]["forward_only_local_update_stability"] == 1.0
    assert report["metrics"]["forward_only_state_budget_integrity"] == 1.0


def test_forward_only_local_update_is_exposed_from_nn_package() -> None:
    import sara_engine.nn as nn

    trace = nn.ForwardOnlyLocalUpdateTrace(capacity=4)
    trace.update(pre_events=[1], post_events=[2], credit=1.0)

    assert trace.read_weight(1, 2) > 0.0
    assert nn.evaluate_forward_only_local_update_trace()["metrics"][
        "forward_only_local_update_stability"
    ] == 1.0
