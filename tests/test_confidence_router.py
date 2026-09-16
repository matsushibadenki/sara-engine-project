import pytest

from sara_engine.learning.confidence_router import ConfidenceRouteConfig, route_prediction


def test_router_overrides_only_when_all_fixed_conditions_pass():
    config = ConfidenceRouteConfig(.60, .20, 4)
    decision = route_prediction(labels=("a", "b", "c"), base_predicted="a",
        base_probabilities={"a": .50, "b": .30, "c": .20}, base_support=8,
        local_predicted="b", local_scores={"a": .1, "b": .5, "c": .2}, config=config)
    assert decision.predicted == "b" and decision.overridden
    assert dict(decision.probabilities) == {"a": .3, "b": .5, "c": .2}


def test_router_retains_base_when_support_or_margin_is_insufficient():
    config = ConfidenceRouteConfig(.60, .40, 4)
    decision = route_prediction(labels=("a", "b"), base_predicted="a",
        base_probabilities={"a": .55, "b": .45}, base_support=3,
        local_predicted="b", local_scores={"a": .1, "b": .3}, config=config)
    assert decision.predicted == "a" and not decision.overridden


def test_router_rejects_incomplete_probability_contract():
    with pytest.raises(ValueError):
        route_prediction(labels=("a", "b"), base_predicted="a", base_probabilities={"a": 1.0},
            base_support=0, local_predicted="b", local_scores={"a": 0.0, "b": 1.0},
            config=ConfidenceRouteConfig(.5, .1, 0))
