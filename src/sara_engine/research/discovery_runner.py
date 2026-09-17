"""Development-only execution boundary for prospective discovery capture."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .discovery_capture import CaptureIntent, CaptureOutcome, DiscoveryCaptureLog


@dataclass(frozen=True)
class CapturedActionResult:
    outcome: CaptureOutcome
    intent_head_sha256: str
    outcome_head_sha256: str


def run_captured_development_action(
    log: DiscoveryCaptureLog,
    intent: CaptureIntent,
    evaluate: Callable[[], CaptureOutcome],
    *,
    expected_head_sha256: str,
) -> CapturedActionResult:
    """Persist intent before evaluation; leave it pending on any execution failure."""
    if not isinstance(log, DiscoveryCaptureLog) or type(intent) is not CaptureIntent:
        raise ValueError("A capture log and intent are required")
    if intent.sequence == 0 or intent.split != "development" or not callable(evaluate):
        raise ValueError("A development action and evaluator are required")
    intent_head = log.begin(intent, expected_head_sha256=expected_head_sha256)
    outcome = evaluate()
    if type(outcome) is not CaptureOutcome or outcome.sequence != intent.sequence or outcome.node_id != intent.node_id:
        raise ValueError("Evaluator outcome does not match the recorded intent")
    outcome_head = log.complete(outcome, expected_head_sha256=intent_head)
    return CapturedActionResult(outcome, intent_head, outcome_head)


def complete_prepared_development_action(
    log: DiscoveryCaptureLog,
    intent: CaptureIntent,
    evaluate: Callable[[], CaptureOutcome],
    *,
    published_intent_head_sha256: str,
) -> CapturedActionResult:
    """Evaluate an already persisted intent only with a caller-published head."""
    if not isinstance(log, DiscoveryCaptureLog) or type(intent) is not CaptureIntent:
        raise ValueError("A capture log and intent are required")
    if intent.sequence == 0 or intent.split != "development" or not callable(evaluate):
        raise ValueError("A development action and evaluator are required")
    view = log.load()
    if view.head_sha256 != published_intent_head_sha256:
        raise ValueError("Published intent head does not match capture")
    if view.pending_node_ids != (intent.node_id,) or view.events[-1] != intent:
        raise ValueError("Published intent is not the sole pending final event")
    outcome = evaluate()
    if type(outcome) is not CaptureOutcome or outcome.sequence != intent.sequence or outcome.node_id != intent.node_id:
        raise ValueError("Evaluator outcome does not match the recorded intent")
    outcome_head = log.complete(outcome, expected_head_sha256=published_intent_head_sha256)
    return CapturedActionResult(outcome, published_intent_head_sha256, outcome_head)
