"""Agent exports with lazy loading for lightweight CPU-only utilities."""

from .bounded_agent_loop import (
    ActionSelectionAblation,
    ActionSelectionArm,
    AgentPlanDecision,
    BoundedAgentLoop,
)

__all__ = [
    "ActionSelectionAblation",
    "ActionSelectionArm",
    "AgentPlanDecision",
    "BoundedAgentLoop",
    "SaraAgent",
    "BoundedTransactionalToolAdapter",
    "ToolStateEdit",
    "TransactionalToolRequest",
    "TransactionalToolResult",
    "CandidateExecution",
    "CandidateExecutionError",
    "IndexedToolCall",
    "IndexedToolResult",
    "IndexedToolResultPairingGate",
    "ToolPairingValidation",
    "BoundedPartialRolloutScheduler",
    "PartialRolloutDispatch",
    "PartialRolloutError",
    "PartialRolloutSliceResult",
    "RolloutResumeContext",
    "IsolatedReversibleToolSandbox",
    "ReversibleToolSandboxError",
    "SandboxCheckpoint",
    "SandboxExecutionResult",
    "SandboxRollbackResult",
]


def __getattr__(name: str):
    if name == "SaraAgent":
        from .sara_agent import SaraAgent

        return SaraAgent
    if name in {
        "BoundedTransactionalToolAdapter",
        "ToolStateEdit",
        "TransactionalToolRequest",
        "TransactionalToolResult",
    }:
        from . import transactional_tools

        return getattr(transactional_tools, name)
    if name in {"CandidateExecution", "CandidateExecutionError"}:
        from . import candidate_execution

        return getattr(candidate_execution, name)
    if name in {
        "IndexedToolCall",
        "IndexedToolResult",
        "IndexedToolResultPairingGate",
        "ToolPairingValidation",
    }:
        from . import tool_result_pairing

        return getattr(tool_result_pairing, name)
    if name in {
        "BoundedPartialRolloutScheduler",
        "PartialRolloutDispatch",
        "PartialRolloutError",
        "PartialRolloutSliceResult",
        "RolloutResumeContext",
    }:
        from . import partial_rollout

        return getattr(partial_rollout, name)
    if name in {
        "IsolatedReversibleToolSandbox",
        "ReversibleToolSandboxError",
        "SandboxCheckpoint",
        "SandboxExecutionResult",
        "SandboxRollbackResult",
    }:
        from . import reversible_tool_sandbox

        return getattr(reversible_tool_sandbox, name)
    raise AttributeError(name)
