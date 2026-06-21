"""Reasoning primitives for sparse, verifiable SARA traces."""

from .sparse_plan_trace import (
    SparsePlanStepResult,
    SparsePlanTraceResult,
    build_repair_materials,
    stable_fact_event_id,
    verify_sparse_plan_trace,
)
from .sparse_reasoning_prior import (
    SparseReasoningPriorResult,
    build_sparse_reasoning_prior,
    evaluate_sparse_reasoning_cases,
    stable_reason_event_id,
)

__all__ = [
    "SparsePlanStepResult",
    "SparsePlanTraceResult",
    "build_repair_materials",
    "stable_fact_event_id",
    "verify_sparse_plan_trace",
    "SparseReasoningPriorResult",
    "build_sparse_reasoning_prior",
    "evaluate_sparse_reasoning_cases",
    "stable_reason_event_id",
]
