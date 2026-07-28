"""Exports public learning-module APIs lazily.

The learning package includes some optional or heavier submodules; lazy exports
keep lightweight callers from importing unrelated dependencies.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple


_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ForceReadout": ("sara_engine.learning.force", "ForceReadout"),
    "export_force_artifact": ("sara_engine.learning.force_io", "export_force_artifact"),
    "load_force_artifact": ("sara_engine.learning.force_io", "load_force_artifact"),
    "build_sine_series": ("sara_engine.learning.force_workflow", "build_sine_series"),
    "evaluate_force_sequence": ("sara_engine.learning.force_workflow", "evaluate_force_sequence"),
    "load_series": ("sara_engine.learning.force_workflow", "load_series"),
    "split_series": ("sara_engine.learning.force_workflow", "split_series"),
    "train_force_sequence": ("sara_engine.learning.force_workflow", "train_force_sequence"),
    "AstroStructuralGateConfig": ("sara_engine.learning.astro_structural_gate", "AstroStructuralGateConfig"),
    "evaluate_astro_structural_gate": ("sara_engine.learning.astro_structural_gate", "evaluate_astro_structural_gate"),
    "DeltaRetentionPolicyConfig": ("sara_engine.learning.delta_retention_policy", "DeltaRetentionPolicyConfig"),
    "build_delta_retention_events": ("sara_engine.learning.delta_retention_policy", "build_delta_retention_events"),
    "evaluate_delta_erase_write_decoupling": ("sara_engine.learning.delta_retention_policy", "evaluate_delta_erase_write_decoupling"),
    "evaluate_delta_retention_policy": ("sara_engine.learning.delta_retention_policy", "evaluate_delta_retention_policy"),
    "evaluate_delta_retention_policy_stress": ("sara_engine.learning.delta_retention_policy", "evaluate_delta_retention_policy_stress"),
    "DendriticGateResult": ("sara_engine.learning.dendritic_feedback", "DendriticGateResult"),
    "SparseDendriticFeedbackGate": ("sara_engine.learning.dendritic_feedback", "SparseDendriticFeedbackGate"),
    "precision_at_expected": ("sara_engine.learning.dendritic_feedback", "precision_at_expected"),
    "DopamineSignalModel": ("sara_engine.learning.reward_modulated_stdp", "DopamineSignalModel"),
    "EligibilityTraceManager": ("sara_engine.learning.reward_modulated_stdp", "EligibilityTraceManager"),
    "RewardModulatedSTDPManager": ("sara_engine.learning.reward_modulated_stdp", "RewardModulatedSTDPManager"),
    "ThreeFactorLearningManager": ("sara_engine.learning.three_factor_learning", "ThreeFactorLearningManager"),
    "GreedyLayerWiseTrainer": ("sara_engine.learning.greedy_layerwise", "GreedyLayerWiseTrainer"),
    "LayerTrainingMetrics": ("sara_engine.learning.greedy_layerwise", "LayerTrainingMetrics"),
    "MetabolicBudgetConfig": ("sara_engine.learning.metabolic_budget", "MetabolicBudgetConfig"),
    "evaluate_structural_metabolic_budget": ("sara_engine.learning.metabolic_budget", "evaluate_structural_metabolic_budget"),
    "MemoryPhaseConfig": ("sara_engine.learning.memory_phase", "MemoryPhaseConfig"),
    "build_memory_phase_observations": ("sara_engine.learning.memory_phase", "build_memory_phase_observations"),
    "evaluate_memory_phase_transitions": ("sara_engine.learning.memory_phase", "evaluate_memory_phase_transitions"),
    "SleepConsolidationConfig": ("sara_engine.learning.sleep_consolidation", "SleepConsolidationConfig"),
    "evaluate_sleep_consolidation": ("sara_engine.learning.sleep_consolidation", "evaluate_sleep_consolidation"),
    "IdleReplayConfig": ("sara_engine.learning.idle_replay", "IdleReplayConfig"),
    "plan_idle_replay": ("sara_engine.learning.idle_replay", "plan_idle_replay"),
    "SynapticTagConfig": ("sara_engine.learning.synaptic_tag", "SynapticTagConfig"),
    "evaluate_synaptic_tags": ("sara_engine.learning.synaptic_tag", "evaluate_synaptic_tags"),
    "OwnLatentPrediction": ("sara_engine.learning.own_latent", "OwnLatentPrediction"),
    "SparseOwnLatentPredictor": ("sara_engine.learning.own_latent", "SparseOwnLatentPredictor"),
    "TokenOverlapBaseline": ("sara_engine.learning.own_latent", "TokenOverlapBaseline"),
    "build_sparse_signature": ("sara_engine.learning.own_latent", "build_sparse_signature"),
    "jaccard_overlap": ("sara_engine.learning.own_latent", "jaccard_overlap"),
    "stable_event_id": ("sara_engine.learning.own_latent", "stable_event_id"),
    "train_predictor_from_cases": ("sara_engine.learning.own_latent", "train_predictor_from_cases"),
    "ResonanceCreditResult": ("sara_engine.learning.resonance_credit", "ResonanceCreditResult"),
    "SparseResonanceCreditAssigner": ("sara_engine.learning.resonance_credit", "SparseResonanceCreditAssigner"),
    "ResonanceEvidenceBundle": ("sara_engine.learning.resonance_evidence", "ResonanceEvidenceBundle"),
    "build_resonance_evidence": ("sara_engine.learning.resonance_evidence", "build_resonance_evidence"),
    "AdaptiveCreditField": ("sara_engine.learning.adaptive_credit", "AdaptiveCreditField"),
    "AdaptiveCreditResult": ("sara_engine.learning.adaptive_credit", "AdaptiveCreditResult"),
    "AdaptiveCreditRouteState": ("sara_engine.learning.adaptive_credit", "AdaptiveCreditRouteState"),
    "summarize_event_memory_credit": ("sara_engine.learning.adaptive_credit", "summarize_event_memory_credit"),
    "BoundedStructuralPlasticityController": ("sara_engine.learning.structural_plasticity", "BoundedStructuralPlasticityController"),
    "StructuralPlasticityManager": ("sara_engine.learning.structural_plasticity", "StructuralPlasticityManager"),
    "StructuralPlasticityResult": ("sara_engine.learning.structural_plasticity", "StructuralPlasticityResult"),
    "StructuralRouteState": ("sara_engine.learning.structural_plasticity", "StructuralRouteState"),
    "RepetitionConsolidationConfig": ("sara_engine.learning.repetition_consolidation", "RepetitionConsolidationConfig"),
    "RepetitionDependentConsolidator": ("sara_engine.learning.repetition_consolidation", "RepetitionDependentConsolidator"),
    "CandidateRepetitionReranker": ("sara_engine.learning.repetition_candidate_reranker", "CandidateRepetitionReranker"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'sara_engine.learning' has no attribute '{name}'")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
