"""Pre-score serialization checks for a future CartPole protocol revision."""
from __future__ import annotations

import json
from operator import index
from typing import Any, Mapping


_AUDIT_KEYS = {
    "environment_id", "gymnasium_version", "python_version", "numpy_version",
    "max_episode_steps", "observation_channels", "events_per_observation",
    "action_count", "evaluator_invoked",
}


def prepare_environment_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only the known NumPy integral field before candidate scoring."""
    if set(audit) != _AUDIT_KEYS:
        raise ValueError("Unexpected environment audit fields")
    if (audit["environment_id"] != "CartPole-v1"
            or audit["max_episode_steps"] != 500
            or audit["observation_channels"] != 4
            or audit["events_per_observation"] != 4
            or audit["evaluator_invoked"] is not False):
        raise ValueError("Environment audit does not match CartPole contract")
    raw_action_count = audit["action_count"]
    if type(raw_action_count) is bool:
        raise ValueError("Action count must be integral")
    try:
        action_count = index(raw_action_count)
    except TypeError as error:
        raise ValueError("Action count must be integral") from error
    if action_count != 2:
        raise ValueError("Unexpected CartPole action count")
    normalized = dict(audit)
    normalized["action_count"] = action_count
    json.dumps(normalized, allow_nan=False)
    return normalized


def preflight_result_envelope(environment_audit: Mapping[str, Any]) -> None:
    """Check the exact outer result shape using representative JSON payloads."""
    audit = prepare_environment_audit(environment_audit)
    representative = {
        "schema": "sara-cartpole-terminal-development-result-v2",
        "protocol_sha256": "0" * 64,
        "environment_audit": audit,
        "runs": [{"run_id": 0, "variants": {"A_scalar_local": {
            "training": [{"steps": 1, "terminal_feedback": 0.002,
                          "input_events": 4, "internal_event_work": 4,
                          "spikes": 0, "terminal_updates": 1,
                          "peak_policy_state_bytes": 1024, "agent_cpu_ns": 1000,
                          "actions_sha256": "0" * 64}],
            "development": [], "final_weights_sha256": "0" * 64,
        }}}],
        "decision": {"passed": False, "heldout_opening_authorized": False},
    }
    json.dumps(representative, sort_keys=True, separators=(",", ":"),
               ensure_ascii=False, allow_nan=False).encode("utf-8")
