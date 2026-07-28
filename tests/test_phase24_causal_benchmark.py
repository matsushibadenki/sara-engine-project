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
    assert report["checks"]["unstable_feedback_freeze"] is True
    assert report["checks"]["branch_records_bounded_and_traceable"] is True
    assert report["checks"]["explicit_rollback_isolated"] is True
    assert report["checks"]["support_paths_and_alternatives_present"] is True


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


def _verified_inference():
    from sara_engine.risa.causal_reasoning import BoundedCausalReasoner, CausalEvidence

    return BoundedCausalReasoner().infer(
        (
            CausalEvidence(
                "switch",
                "light",
                "causes_candidate",
                "fixture:branch",
                intervention_count=2,
                contrastive_count=2,
                confidence=0.95,
                event_path=("event:switch", "event:light"),
                context_tags=("room:a",),
            ),
        )
    )


def test_counterfactual_branch_records_preserve_context_and_rollback_explicitly():
    from sara_engine.risa.causal_reasoning import BoundedCausalReasoner

    reasoner = BoundedCausalReasoner()
    staged = reasoner.branch_counterfactual(
        _verified_inference(),
        intervention="remove_switch",
        predicted_outcome="light_on",
        alternative_outcome="light_off",
        depth=2,
        context_tags=("trial:1",),
    )
    rolled_back = reasoner.rollback_counterfactual(staged, reason="trial_complete")

    assert staged.abstained is False
    assert len(staged.branches) == 2
    assert all(item.status == "staged" for item in staged.branches)
    assert all("room:a" in item.context_tags for item in staged.branches)
    assert all("trial:1" in item.context_tags for item in staged.branches)
    assert all(item.supporting_paths for item in staged.branches)
    assert rolled_back.rolled_back is True
    assert rolled_back.rollback_reason == "trial_complete"
    assert all(item.status == "rolled_back" for item in rolled_back.branches)
    assert all(item.status == "staged" for item in staged.branches)


def test_counterfactual_branch_records_enforce_depth_event_and_state_budgets():
    from sara_engine.risa.causal_reasoning import BoundedCausalReasoner

    inference = _verified_inference()
    depth_blocked = BoundedCausalReasoner(max_branch_depth=1).branch_counterfactual(
        inference,
        intervention="remove",
        predicted_outcome="on",
        alternative_outcome="off",
        depth=2,
    )
    event_blocked = BoundedCausalReasoner(
        max_branch_event_cost=1
    ).branch_counterfactual(
        inference,
        intervention="remove",
        predicted_outcome="on",
        alternative_outcome="off",
    )
    state_blocked = BoundedCausalReasoner(
        max_branch_state_bytes=256
    ).branch_counterfactual(
        inference,
        intervention="remove",
        predicted_outcome="on",
        alternative_outcome="off",
    )

    assert depth_blocked.abstained is True
    assert depth_blocked.branches == ()
    assert event_blocked.reason == "branch_event_or_state_budget_exceeded"
    assert event_blocked.branches == ()
    assert state_blocked.reason == "branch_event_or_state_budget_exceeded"
    assert state_blocked.branches == ()


def test_unstable_feedback_freezes_causal_promotion_with_alternative_action():
    from sara_engine.risa.causal_reasoning import BoundedCausalReasoner, CausalEvidence

    inference = BoundedCausalReasoner().infer(
        (
            CausalEvidence(
                "valve",
                "flow",
                "causes_candidate",
                "fixture:unstable",
                intervention_count=2,
                contrastive_count=2,
                confidence=0.95,
                feedback_stable=False,
            ),
        )
    )

    assert inference.abstained is True
    assert inference.reason == "unstable_feedback_freeze"
    assert "collect_stable_feedback_revision" in inference.alternatives
