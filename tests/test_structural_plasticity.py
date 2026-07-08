import importlib.util
import os
import sys


def _load_module():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "sara_engine",
            "learning",
            "structural_plasticity.py",
        )
    )
    spec = importlib.util.spec_from_file_location("structural_plasticity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BoundedStructuralPlasticityController = _load_module().BoundedStructuralPlasticityController


def test_structural_plasticity_promotes_verified_provisional_route_to_stable() -> None:
    controller = BoundedStructuralPlasticityController(
        min_stable_verified_support=2,
        min_stable_prediction_gain=0.6,
        max_rewrites_per_event=2,
    )
    route = (1, 2)
    controller.register_route(
        route,
        weight=0.4,
        route_state="provisional",
        responsibility=0.35,
        prediction_gain_support=0.35,
        support_count=1,
        verified_support_count=1,
        created_step=0,
        last_active_step=0,
    )

    result = controller.apply_event(
        active_routes={route: 0.9},
        signals={
            "prediction_error": 0.72,
            "novelty": 0.30,
            "reward": 0.64,
            "contradiction": 0.05,
            "metabolic_headroom": 0.90,
            "source_backed": True,
        },
        event_memory_support={
            route: {
                "prediction_gain_support": 0.45,
                "replay_support": 0.60,
                "verified": True,
            }
        },
    )

    assert result.update_allowed is True
    assert route in result.stabilized_routes
    assert controller.routes[route].route_state == "stable"
    assert controller.routes[route].verified_support_count >= 2


def test_structural_plasticity_marks_stale_route_decaying_then_prunes_it() -> None:
    controller = BoundedStructuralPlasticityController(
        prune_grace_steps=2,
        min_active_responsibility=0.25,
        contradiction_prune_threshold=1,
        max_rewrites_per_event=2,
    )
    route = (4, 5)
    controller.register_route(
        route,
        weight=0.2,
        route_state="provisional",
        responsibility=0.05,
        contradiction_count=1,
        created_step=-2,
        last_active_step=-2,
    )

    first = controller.apply_event(
        active_routes={},
        signals={
            "prediction_error": 0.80,
            "novelty": 0.10,
            "reward": 0.10,
            "contradiction": 0.70,
            "metabolic_headroom": 0.90,
            "source_backed": True,
        },
    )
    second = controller.apply_event(
        active_routes={},
        signals={
            "prediction_error": 0.82,
            "novelty": 0.10,
            "reward": 0.10,
            "contradiction": 0.72,
            "metabolic_headroom": 0.90,
            "source_backed": True,
        },
    )

    assert route in first.decaying_routes
    assert route not in first.pruned_routes
    assert route in second.pruned_routes
    assert route not in controller.routes


def test_structural_plasticity_grows_verified_candidate_within_budget() -> None:
    controller = BoundedStructuralPlasticityController(
        max_total_links=2,
        max_fan_in=2,
        max_fan_out=2,
        provisional_growth_threshold=0.7,
    )

    result = controller.apply_event(
        active_routes={(1, 1): 0.8},
        signals={
            "prediction_error": 0.74,
            "novelty": 0.30,
            "reward": 0.20,
            "contradiction": 0.05,
            "metabolic_headroom": 0.90,
            "source_backed": True,
        },
        candidate_routes={
            (3, 7): {
                "coactivation": 0.76,
                "prediction_gain_support": 0.73,
                "verified": True,
                "weight": 0.18,
            }
        },
    )

    assert result.update_allowed is True
    assert (3, 7) in result.grown_routes
    assert controller.routes[(3, 7)].route_state == "provisional"


def test_structural_plasticity_blocks_growth_during_frozen_evaluation() -> None:
    controller = BoundedStructuralPlasticityController()

    result = controller.apply_event(
        active_routes={},
        signals={
            "prediction_error": 0.90,
            "novelty": 0.60,
            "reward": 0.60,
            "contradiction": 0.05,
            "metabolic_headroom": 0.90,
            "source_backed": True,
        },
        candidate_routes={
            (8, 9): {
                "coactivation": 0.95,
                "prediction_gain_support": 0.91,
                "verified": True,
            }
        },
        frozen_evaluation=True,
    )

    assert result.update_allowed is False
    assert result.decision == "freeze_evaluation"
    assert result.grown_routes == ()
    assert (8, 9) not in controller.routes
