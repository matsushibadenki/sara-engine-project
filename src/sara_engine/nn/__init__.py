"""Public API for sara_engine.nn with lazy symbol loading."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "SNNModule": ("sara_engine.nn.module", "SNNModule"),
    "Sequential": ("sara_engine.nn.sequential", "Sequential"),
    "LinearSpike": ("sara_engine.nn.linear_spike", "LinearSpike"),
    "SpikeSelfAttention": ("sara_engine.nn.attention", "SpikeSelfAttention"),
    "PredictiveSpikeLayer": ("sara_engine.nn.predictive", "PredictiveSpikeLayer"),
    "SpikingPredictiveLayer": ("sara_engine.nn.predictive", "SpikingPredictiveLayer"),
    "SpikeDropout": ("sara_engine.nn.dropout", "SpikeDropout"),
    "SpikeLayerNorm": ("sara_engine.nn.normalization", "SpikeLayerNorm"),
    "CrossModalAssociator": ("sara_engine.nn.multimodal", "CrossModalAssociator"),
    "RewardModulatedLinearSpike": ("sara_engine.nn.rstdp", "RewardModulatedLinearSpike"),
    "UnsupervisedSpikeLayer": ("sara_engine.nn.unsupervised_layer", "UnsupervisedSpikeLayer"),
    "SpatioTemporalBindingLayer": (
        "sara_engine.nn.spatio_temporal_binding",
        "SpatioTemporalBindingLayer",
    ),
    "LocalManifoldTransitionMemory": (
        "sara_engine.nn.local_manifold_memory",
        "LocalManifoldTransitionMemory",
    ),
    "DeltaAssociativeSpikeMemory": (
        "sara_engine.nn.delta_associative_memory",
        "DeltaAssociativeSpikeMemory",
    ),
    "evaluate_delta_memory_steering_trace": (
        "sara_engine.nn.delta_associative_memory",
        "evaluate_delta_memory_steering_trace",
    ),
    "MultiTimescaleLeakState": (
        "sara_engine.nn.multi_timescale_leak_state",
        "MultiTimescaleLeakState",
    ),
    "evaluate_multi_timescale_leak_state": (
        "sara_engine.nn.multi_timescale_leak_state",
        "evaluate_multi_timescale_leak_state",
    ),
    "SparseLiquidTimeConstantNeuron": (
        "sara_engine.nn.sparse_liquid_time_constant",
        "SparseLiquidTimeConstantNeuron",
    ),
    "SparseLiquidTrace": (
        "sara_engine.nn.sparse_liquid_time_constant",
        "SparseLiquidTrace",
    ),
    "PhaseSynchronizedBindingTrace": (
        "sara_engine.nn.phase_synchronized_binding_trace",
        "PhaseSynchronizedBindingTrace",
    ),
    "evaluate_phase_synchronized_binding_trace": (
        "sara_engine.nn.phase_synchronized_binding_trace",
        "evaluate_phase_synchronized_binding_trace",
    ),
    "ForwardOnlyLocalUpdateTrace": (
        "sara_engine.nn.forward_only_local_update",
        "ForwardOnlyLocalUpdateTrace",
    ),
    "evaluate_forward_only_local_update_trace": (
        "sara_engine.nn.forward_only_local_update",
        "evaluate_forward_only_local_update_trace",
    ),
    "PlasticSubmodelRegistry": (
        "sara_engine.nn.plastic_submodel_registry",
        "PlasticSubmodelRegistry",
    ),
    "build_default_plastic_submodel_registry": (
        "sara_engine.nn.plastic_submodel_registry",
        "build_default_plastic_submodel_registry",
    ),
    "evaluate_plastic_submodel_registry_trace": (
        "sara_engine.nn.plastic_submodel_registry",
        "evaluate_plastic_submodel_registry_trace",
    ),
    "evaluate_plastic_submodel_intervention_trace": (
        "sara_engine.nn.plastic_submodel_registry",
        "evaluate_plastic_submodel_intervention_trace",
    ),
    "evaluate_plastic_submodel_credit_assignment_trace": (
        "sara_engine.nn.plastic_submodel_registry",
        "evaluate_plastic_submodel_credit_assignment_trace",
    ),
    "evaluate_plastic_submodel_structural_adaptation_trace": (
        "sara_engine.nn.plastic_submodel_registry",
        "evaluate_plastic_submodel_structural_adaptation_trace",
    ),
    "evaluate_plastic_submodel_scientific_model_trace": (
        "sara_engine.nn.plastic_submodel_registry",
        "evaluate_plastic_submodel_scientific_model_trace",
    ),
    "evaluate_plastic_submodel_open_ended_hypothesis_bank_trace": (
        "sara_engine.nn.plastic_submodel_registry",
        "evaluate_plastic_submodel_open_ended_hypothesis_bank_trace",
    ),
    "SparseVerifier": (
        "sara_engine.nn.sparse_verifier",
        "SparseVerifier",
    ),
    "SparseVerifierThresholds": (
        "sara_engine.nn.sparse_verifier",
        "SparseVerifierThresholds",
    ),
    "evaluate_sparse_verifier_trace": (
        "sara_engine.nn.sparse_verifier",
        "evaluate_sparse_verifier_trace",
    ),
    "evaluate_sparse_best_of_n_trace": (
        "sara_engine.nn.sparse_verifier",
        "evaluate_sparse_best_of_n_trace",
    ),
    "evaluate_self_correction_trace": (
        "sara_engine.nn.sparse_verifier",
        "evaluate_self_correction_trace",
    ),
    "evaluate_bounded_tree_search_trace": (
        "sara_engine.nn.sparse_verifier",
        "evaluate_bounded_tree_search_trace",
    ),
    "evaluate_reasoning_forest_lane_trace": (
        "sara_engine.nn.sparse_verifier",
        "evaluate_reasoning_forest_lane_trace",
    ),
    "evaluate_hierarchical_reasoning_trace": (
        "sara_engine.nn.sparse_verifier",
        "evaluate_hierarchical_reasoning_trace",
    ),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'sara_engine.nn' has no attribute '{name}'")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
