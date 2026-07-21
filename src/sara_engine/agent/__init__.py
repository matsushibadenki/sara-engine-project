"""Agent exports with lazy loading for lightweight CPU-only utilities."""

from .bounded_agent_loop import AgentPlanDecision, BoundedAgentLoop

__all__ = ["AgentPlanDecision", "BoundedAgentLoop", "SaraAgent"]


def __getattr__(name: str):
    if name == "SaraAgent":
        from .sara_agent import SaraAgent

        return SaraAgent
    raise AttributeError(name)
