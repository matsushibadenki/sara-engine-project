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
    raise AttributeError(name)
