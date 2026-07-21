from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "phase24_causal_benchmark.py"
    spec = importlib.util.spec_from_file_location("phase24_causal_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase24_causal_benchmark_passes():
    module = _load_module()
    fixture = PROJECT_ROOT / "data" / "processed" / "benchmark_fixtures" / "phase24_causal_cases.jsonl"
    report = module.build_report(module._load(str(fixture)))

    assert report["passed"] is True
    assert report["checks"]["temporal_order_not_verified_causal"] is True
    assert report["checks"]["durable_mutation_blocked"] is True
    assert report["checks"]["event_memory_only_verified_causal"] is True


def test_phase24_reasoner_requires_intervention_and_contrastive_support():
    from sara_engine.risa.causal_reasoning import BoundedCausalReasoner, CausalEvidence

    inference = BoundedCausalReasoner().infer(
        (CausalEvidence("a", "b", "causes_candidate", "fixture", intervention_count=1, contrastive_count=0, confidence=0.9),)
    )

    assert inference.relation_type == "causes_candidate"
    assert BoundedCausalReasoner().counterfactual(
        inference,
        intervention="remove",
        predicted_outcome="b",
        alternative_outcome="not_b",
    )["abstained"] is True


def test_phase24_promotes_candidate_from_intervention_and_contrastive_evidence():
    from sara_engine.risa.causal_reasoning import BoundedCausalReasoner, CausalEvidence

    inference = BoundedCausalReasoner().infer(
        (
            CausalEvidence(
                "switch",
                "light",
                "causes_candidate",
                "fixture:intervention",
                intervention_count=2,
                contrastive_count=2,
                confidence=0.9,
            ),
        )
    )

    assert inference.relation_type == "causes_verified"


def test_causal_candidate_is_not_verified_for_event_memory():
    from sara_engine.risa.causal_reasoning import BoundedCausalReasoner, CausalEvidence, causal_event_state_candidate
    from sara_engine.memory.event_state_cache import VerifiedHierarchicalEventStateCache

    inference = BoundedCausalReasoner().infer(
        (CausalEvidence("a", "b", "precedes", "fixture", confidence=0.9),)
    )
    admission = VerifiedHierarchicalEventStateCache().admit(
        causal_event_state_candidate(inference, source_ref="fixture")
    )

    assert admission.accepted is False
    assert admission.decision == "block_failed_verification"
