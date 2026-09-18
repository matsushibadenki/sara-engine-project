"""One-shot development evaluator for the terminal-feedback CartPole study.

Held-out evaluation is intentionally absent. Run only against a separately
frozen, source-pinned preregistration; never tune it after seeing outcomes.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import process_time_ns
from typing import Any

from sara_engine.evaluation.cartpole_local_policy import CartPoleLocalPolicy
from sara_engine.evaluation.cartpole_sparse_reward import (
    ENVIRONMENT_ID, MAX_EPISODE_STEPS, encode_observation, inspect_environment,
    terminal_feedback,
)
from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.utils.project_paths import ensure_allowed_output_path, project_path


PROTOCOL_PATH = "data/processed/benchmark_fixtures/cartpole_terminal_prereg_v1.json"
VARIANTS = (
    "A_scalar_local", "B_compact_event", "C_stateful_spiking",
    "C_spike_bypass", "C_state_reset", "A_no_learning",
)
PINNED_SOURCES = {
    "src/sara_engine/evaluation/cartpole_sparse_reward.py",
    "src/sara_engine/evaluation/cartpole_local_policy.py",
    "src/sara_engine/evaluation/cartpole_registered_runner.py",
    "src/sara_engine/evaluation/event_unit_causal_isolation.py",
    "src/sara_engine/evaluation/r1_temporal_learning_benchmark.py",
    "src/sara_engine/neuro/neuron.py",
}


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def _source_sha(path: str) -> str:
    return sha256(Path(project_path(path)).read_bytes()).hexdigest()


def load_frozen_protocol(expected_sha256: str) -> dict:
    """Reject all unpinned, mutated, or structurally unexpected protocols."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("A frozen preregistration digest is required")
    raw = Path(project_path(PROTOCOL_PATH)).read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Preregistration digest mismatch")
    protocol = json.loads(raw)
    if raw != _canonical_bytes(protocol) or protocol.get("schema") != "sara-cartpole-terminal-prereg-v1":
        raise ValueError("Noncanonical or unsupported preregistration")
    if protocol.get("task_sha256") != _source_sha(
        "data/processed/benchmark_fixtures/cartpole_terminal_task_v1.json"
    ):
        raise ValueError("Task contract changed")
    if set(protocol["source_sha256"]) != PINNED_SOURCES:
        raise ValueError("Pinned source set changed")
    for path, digest in protocol["source_sha256"].items():
        if _source_sha(path) != digest:
            raise ValueError("Pinned evaluator source changed")
    if tuple(protocol["variants"]) != VARIANTS:
        raise ValueError("Variant order changed")
    if protocol["policy_seed_formula"] != "410000 + environment_reset_seed":
        raise ValueError("Policy seed formula changed")
    if protocol["runs"] != [0, 1, 2, 3, 4] or protocol["train_episodes"] != 128 \
            or protocol["development_episodes"] != 32:
        raise ValueError("Training or development budget changed")
    if protocol["development_only"] is not True or protocol["heldout_opening_authorized"] is not False:
        raise ValueError("Held-out partition is not authorized")
    if protocol["development_scoring_authorized"] is not True:
        raise ValueError("Development scoring has not been registered")
    if protocol["development_output"] != "workspace/evaluation/cartpole_terminal_development_v1.json":
        raise ValueError("Development result location changed")
    if protocol["resource_ceiling"]["peak_policy_state_bytes"] != 200000 \
            or protocol["resource_ceiling"]["agent_cpu_ns_per_episode"] != 2_000_000_000:
        raise ValueError("Resource ceiling changed")
    if protocol["gate"] != {
        "minimum_C_mean_steps": 100.0,
        "minimum_C_minus_B_mean_steps": 20.0,
        "minimum_C_minus_bypass_mean_steps": 20.0,
        "minimum_C_minus_reset_mean_steps": 10.0,
        "minimum_C_minus_no_learning_mean_steps": 20.0,
        "minimum_positive_C_minus_B_runs": 4,
        "require_A_B_identical_actions_and_weights": True,
        "require_all_resource_ceilings": True,
    }:
        raise ValueError("Development decision gate changed")
    return protocol


def _make_policy(variant: str, run_id: int) -> CartPoleLocalPolicy:
    if variant == "C_spike_bypass":
        return CartPoleLocalPolicy("C_stateful_spiking", 410000 + run_id,
                                   spike_bypass=True)
    if variant == "C_state_reset":
        return CartPoleLocalPolicy("C_stateful_spiking", 410000 + run_id,
                                   state_reset_each_step=True)
    arm = "A_scalar_local" if variant == "A_no_learning" else variant
    return CartPoleLocalPolicy(arm, 410000 + run_id)


def _run_episode(env: Any, policy: CartPoleLocalPolicy, *, seed: int,
                 learn: bool) -> dict:
    observation, _ = env.reset(seed=seed)
    before_work, before_spikes, before_updates = (
        policy.event_work, policy.spikes, policy.terminal_updates,
    )
    start = process_time_ns()
    policy.begin_episode(random_seed=410000 + seed)
    peak_state = _deep_size(policy)
    cpu_ns = process_time_ns() - start
    action_trace = bytearray()
    for step_index in range(MAX_EPISODE_STEPS):
        start = process_time_ns()
        events = encode_observation(observation, step_index)
        action = policy.choose(events)
        peak_state = max(peak_state, _deep_size(policy))
        cpu_ns += process_time_ns() - start
        action_trace.append(action)
        observation, _default_reward, terminated, truncated, _ = env.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError("Invalid environment terminal flags")
        if terminated or truncated:
            start = process_time_ns()
            outcome = terminal_feedback(steps=step_index + 1,
                                        terminated=terminated, truncated=truncated)
            policy.finish_episode(outcome, learn=learn)
            cpu_ns += process_time_ns() - start
            return {
                "steps": step_index + 1,
                "terminal_feedback": outcome,
                "input_events": 4 * (step_index + 1),
                "internal_event_work": policy.event_work - before_work,
                "spikes": policy.spikes - before_spikes,
                "terminal_updates": policy.terminal_updates - before_updates,
                "peak_policy_state_bytes": peak_state,
                "agent_cpu_ns": cpu_ns,
                "actions_sha256": sha256(action_trace).hexdigest(),
            }
    raise ValueError("CartPole failed to terminate within 500 steps")


def _resource_ok(episodes: list[dict], variant: str) -> bool:
    for item in episodes:
        steps = item["steps"]
        maximum_work = (steps * (44 if variant == "C_state_reset" else
                                 24 if variant == "C_stateful_spiking" else 4)
                        + (20 if variant == "C_stateful_spiking" else 0) + 40)
        if (item["input_events"] != steps * 4
                or item["internal_event_work"] > maximum_work
                or item["peak_policy_state_bytes"] > 200000
                or item["agent_cpu_ns"] > 2_000_000_000):
            return False
    return True


def development_decision(runs: list[dict]) -> dict:
    """Apply the frozen conjunction; failures never authorize held-out use."""
    if len(runs) != 5 or [run["run_id"] for run in runs] != list(range(5)):
        raise ValueError("Five ordered runs are required")
    means = {}
    resource_ok = True
    parity_ok = True
    positive_runs = 0
    for variant in VARIANTS:
        means[variant] = sum(sum(e["steps"] for e in run["variants"][variant]["development"])
                             for run in runs) / (5 * 32)
    for run in runs:
        variants = run["variants"]
        for variant in VARIANTS:
            record = variants[variant]
            if len(record["training"]) != 128 or len(record["development"]) != 32:
                raise ValueError("Episode budget mismatch")
            resource_ok &= _resource_ok(record["training"] + record["development"], variant)
            if variant == "A_no_learning":
                parity_ok &= all(e["terminal_updates"] == 0 for e in record["training"])
            parity_ok &= all(e["terminal_updates"] == 0 for e in record["development"])
        a, b = variants["A_scalar_local"], variants["B_compact_event"]
        parity_ok &= ([(e["steps"], e["actions_sha256"]) for e in a["training"] + a["development"]]
                      == [(e["steps"], e["actions_sha256"]) for e in b["training"] + b["development"]]
                      and a["final_weights_sha256"] == b["final_weights_sha256"])
        c_mean = sum(e["steps"] for e in variants["C_stateful_spiking"]["development"]) / 32
        b_mean = sum(e["steps"] for e in b["development"]) / 32
        positive_runs += c_mean > b_mean
    c = means["C_stateful_spiking"]
    checks = {
        "C_mean_at_least_100": c >= 100.0,
        "C_minus_B_at_least_20": c - means["B_compact_event"] >= 20.0,
        "C_minus_bypass_at_least_20": c - means["C_spike_bypass"] >= 20.0,
        "C_minus_reset_at_least_10": c - means["C_state_reset"] >= 10.0,
        "C_minus_no_learning_at_least_20": c - means["A_no_learning"] >= 20.0,
        "C_minus_B_positive_on_four_runs": positive_runs >= 4,
        "A_B_exact_parity_and_no_dev_updates": bool(parity_ok),
        "all_resource_ceilings": bool(resource_ok),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "development_mean_steps": means, "positive_C_minus_B_runs": positive_runs,
            "heldout_opening_authorized": False}


def _reserve_one_shot(output: Path, protocol_sha256: str) -> None:
    """Retain the reservation even if execution fails; retry requires review."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output) + ".lock").open("xb") as stream:
        stream.write((protocol_sha256 + "\n").encode("ascii"))


def run_registered_development(*, expected_protocol_sha256: str) -> dict:
    """Score the registered development split once; never open held-out here."""
    protocol = load_frozen_protocol(expected_protocol_sha256)
    resolved_output = Path(ensure_allowed_output_path(protocol["development_output"]))
    if resolved_output.exists():
        raise ValueError("Development output must be a new managed evaluation file")
    if resolved_output.suffix != ".json":
        raise ValueError("Development output must be JSON")
    environment_audit = inspect_environment()
    _reserve_one_shot(resolved_output, expected_protocol_sha256)
    import gymnasium as gym

    runs = []
    for run_id in protocol["runs"]:
        variants = {}
        for variant in VARIANTS:
            policy = _make_policy(variant, run_id)
            env = gym.make(ENVIRONMENT_ID)
            try:
                training = [_run_episode(env, policy, seed=110000 + 1000 * run_id + index,
                                         learn=variant != "A_no_learning")
                            for index in range(128)]
                development = [_run_episode(env, policy, seed=210000 + 1000 * run_id + index,
                                            learn=False)
                               for index in range(32)]
            finally:
                env.close()
            weights_digest = sha256(_canonical_bytes(policy.weight_snapshot())).hexdigest()
            variants[variant] = {"training": training, "development": development,
                                 "final_weights_sha256": weights_digest}
        runs.append({"run_id": run_id, "variants": variants})
    decision = development_decision(runs)
    result = {
        "schema": "sara-cartpole-terminal-development-result-v1",
        "protocol_sha256": expected_protocol_sha256,
        "environment_audit": environment_audit,
        "runs": runs,
        "decision": decision,
    }
    with resolved_output.open("xb") as stream:
        stream.write(_canonical_bytes(result))
    return result
