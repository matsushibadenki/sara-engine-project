from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromotionPolicy:
    mode: str
    min_corpus_lines: int
    require_operational_script: bool


def resolve_policy(mode: str) -> PromotionPolicy:
    normalized = (mode or "balanced").strip().lower()
    if normalized == "strict":
        return PromotionPolicy(
            mode="strict",
            min_corpus_lines=500,
            require_operational_script=True,
        )
    if normalized == "exploratory":
        return PromotionPolicy(
            mode="exploratory",
            min_corpus_lines=50,
            require_operational_script=False,
        )
    return PromotionPolicy(
        mode="balanced",
        min_corpus_lines=120,
        require_operational_script=False,
    )


def can_promote(eval_report: dict[str, object], policy: PromotionPolicy) -> tuple[bool, str]:
    corpus_lines = int(eval_report.get("corpus_lines", 0) or 0)
    if corpus_lines < policy.min_corpus_lines:
        return False, f"corpus_lines_below_policy:{corpus_lines}<{policy.min_corpus_lines}"

    if policy.require_operational_script:
        op = eval_report.get("operational_readiness", {})
        if not isinstance(op, dict) or not bool(op.get("passed", False)):
            return False, "operational_script_required"

    if not bool(eval_report.get("passed", False)):
        return False, "evaluation_failed"

    return True, "ok"
