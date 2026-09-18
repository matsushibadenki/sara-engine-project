"""Fresh-seed V2 CartPole development runner after V1 serialization failure.

The V1 policy, episode evaluator, variants, and decision gate are reused.
Only split identity and result-envelope serialization are corrected.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from sara_engine.evaluation import cartpole_registered_runner as v1
from sara_engine.evaluation.cartpole_result_preflight import (
    prepare_environment_audit, preflight_result_envelope,
)
from sara_engine.evaluation.cartpole_sparse_reward import ENVIRONMENT_ID, inspect_environment
from sara_engine.utils.project_paths import ensure_allowed_output_path, project_path


V1_PROTOCOL_SHA256 = "86a50d698174c220afa66bac25160caf293762fbfe7875d56a80a9efea51f392"
PROTOCOL_PATH = "data/processed/benchmark_fixtures/cartpole_terminal_prereg_v2.json"
TASK_PATH = "data/processed/benchmark_fixtures/cartpole_terminal_task_v2.json"
SPLIT_BASES = {"training": 510000, "development": 610000, "heldout": 710000}
OUTPUT_PATH = "workspace/evaluation/cartpole_terminal_development_v2.json"
PINNED_SOURCES = v1.PINNED_SOURCES | {
    "src/sara_engine/evaluation/cartpole_registered_runner_v2.py",
    "src/sara_engine/evaluation/cartpole_result_preflight.py",
}


def load_frozen_protocol_v2(expected_sha256: str) -> dict:
    """Verify the fresh split and unchanged V1 scientific decision rule."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("A V2 preregistration digest is required")
    raw = Path(project_path(PROTOCOL_PATH)).read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("V2 preregistration digest mismatch")
    protocol = json.loads(raw)
    if (raw != v1._canonical_bytes(protocol)
            or protocol.get("schema") != "sara-cartpole-terminal-prereg-v2"):
        raise ValueError("V2 preregistration is not canonical")
    parent = v1.load_frozen_protocol(V1_PROTOCOL_SHA256)
    if protocol["parent_protocol_sha256"] != V1_PROTOCOL_SHA256:
        raise ValueError("V1 parent identity changed")
    task_raw = Path(project_path(TASK_PATH)).read_bytes()
    if sha256(task_raw).hexdigest() != protocol["task_sha256"]:
        raise ValueError("V2 task contract changed")
    task = json.loads(task_raw)
    if (task.get("schema") != "sara-cartpole-terminal-task-contract-v2"
            or task.get("parent_task_sha256") != parent["task_sha256"]
            or task.get("environment_id") != ENVIRONMENT_ID
            or task.get("unchanged_interface_source_sha256") !=
            parent["source_sha256"]["src/sara_engine/evaluation/cartpole_sparse_reward.py"]
            or task.get("split_bases") != SPLIT_BASES
            or task.get("run_ids") != [0, 1, 2, 3, 4]
            or task.get("episodes_per_run") != {
                "training": 128, "development": 32, "heldout": 32,
            }
            or task.get("heldout_sealed") is not True
            or task.get("development_policy_updates") is not False
            or task.get("heldout_policy_updates") is not False
            or task.get("default_step_reward_used_for_learning") is not False):
        raise ValueError("V2 task split changed")
    if (set(protocol["source_sha256"]) != PINNED_SOURCES
            or any(v1._source_sha(path) != digest
                   for path, digest in protocol["source_sha256"].items())):
        raise ValueError("V2 pinned source changed")
    if (protocol["variants"] != parent["variants"]
            or protocol["gate"] != parent["gate"]
            or protocol["resource_ceiling"] != parent["resource_ceiling"]
            or protocol["runs"] != parent["runs"]
            or protocol["train_episodes"] != parent["train_episodes"]
            or protocol["development_episodes"] != parent["development_episodes"]
            or protocol["policy_seed_formula"] != parent["policy_seed_formula"]):
        raise ValueError("V1 learner or decision budget changed")
    if (protocol["split_bases"] != SPLIT_BASES
            or protocol["development_output"] != OUTPUT_PATH
            or protocol["development_only"] is not True
            or protocol["development_scoring_authorized"] is not True
            or protocol["heldout_opening_authorized"] is not False
            or protocol["serialization_preflight_required"] is not True):
        raise ValueError("V2 execution boundaries changed")
    return protocol


def run_registered_development_v2(*, expected_protocol_sha256: str) -> dict:
    """Score V2 development exactly once; no held-out path exists here."""
    protocol = load_frozen_protocol_v2(expected_protocol_sha256)
    output = Path(ensure_allowed_output_path(protocol["development_output"]))
    if output.exists() or output.suffix != ".json":
        raise ValueError("V2 development output is unavailable")
    raw_audit = inspect_environment()
    preflight_result_envelope(raw_audit)
    environment_audit = prepare_environment_audit(raw_audit)
    v1._canonical_bytes({
        "schema": "sara-cartpole-terminal-development-result-v2",
        "protocol_sha256": expected_protocol_sha256,
        "environment_audit": environment_audit,
        "runs": [], "decision": {},
    })
    v1._reserve_one_shot(output, expected_protocol_sha256)
    import gymnasium as gym

    runs = []
    for run_id in protocol["runs"]:
        variants = {}
        for variant in v1.VARIANTS:
            policy = v1._make_policy(variant, run_id)
            env = gym.make(ENVIRONMENT_ID)
            try:
                training = [v1._run_episode(
                    env, policy, seed=SPLIT_BASES["training"] + 1000 * run_id + index,
                    learn=variant != "A_no_learning",
                ) for index in range(protocol["train_episodes"])]
                development = [v1._run_episode(
                    env, policy, seed=SPLIT_BASES["development"] + 1000 * run_id + index,
                    learn=False,
                ) for index in range(protocol["development_episodes"])]
            finally:
                env.close()
            variants[variant] = {
                "training": training,
                "development": development,
                "final_weights_sha256": sha256(
                    v1._canonical_bytes(policy.weight_snapshot())
                ).hexdigest(),
            }
        runs.append({"run_id": run_id, "variants": variants})
    result = {
        "schema": "sara-cartpole-terminal-development-result-v2",
        "protocol_sha256": expected_protocol_sha256,
        "environment_audit": environment_audit,
        "runs": runs,
        "decision": v1.development_decision(runs),
    }
    payload = v1._canonical_bytes(result)
    with output.open("xb") as stream:
        stream.write(payload)
    return result
