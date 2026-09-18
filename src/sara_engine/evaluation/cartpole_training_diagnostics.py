"""Read-only, training-only diagnostics for the frozen CartPole V2 result."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

from sara_engine.evaluation.cartpole_registered_runner import VARIANTS
from sara_engine.neuro.neuron import Neuron
from sara_engine.utils.project_paths import workspace_path


RESULT_SHA256 = "866c7c35a1bb39d61818fe779b636f44353877a215848f6f02f7412ce82ef63e"
PROTOCOL_SHA256 = "1da4356e7eb88627a1fe62390f2a7bfdb3f06047506693043fcf03b6b67132bd"
RESULT_PATH = workspace_path("evaluation", "cartpole_terminal_development_v2.json")


def load_training_records() -> list[dict]:
    """Return only training episodes; never expose development records."""
    raw = Path(RESULT_PATH).read_bytes()
    if sha256(raw).hexdigest() != RESULT_SHA256:
        raise ValueError("Frozen CartPole result digest changed")
    result = json.loads(raw)
    if (result.get("schema") != "sara-cartpole-terminal-development-result-v2"
            or result.get("protocol_sha256") != PROTOCOL_SHA256
            or result.get("decision", {}).get("passed") is not False):
        raise ValueError("Unexpected CartPole result identity or decision")
    runs = result["runs"]
    if len(runs) != 5 or [run["run_id"] for run in runs] != list(range(5)):
        raise ValueError("Unexpected CartPole run set")
    training_only = []
    for run in runs:
        variants = run["variants"]
        if set(variants) != set(VARIANTS):
            raise ValueError("Unexpected CartPole variant set")
        selected = {}
        for name in VARIANTS:
            episodes = variants[name]["training"]
            if len(episodes) != 128:
                raise ValueError("Unexpected training episode count")
            for episode in episodes:
                steps = episode["steps"]
                if (type(steps) is not int or not 1 <= steps <= 500
                        or episode["input_events"] != 4 * steps
                        or episode["terminal_feedback"] != steps / 500):
                    raise ValueError("Training episode accounting changed")
            selected[name] = episodes
        training_only.append({"run_id": run["run_id"], "training": selected})
    return training_only


def summarize_training(records: list[dict]) -> list[dict]:
    """Summarize fixed episode quartiles without inspecting development data."""
    summaries = []
    for run in records:
        conditions = {}
        for name, episodes in run["training"].items():
            if len(episodes) != 128:
                raise ValueError("Exactly 128 training episodes are required")
            inputs = sum(episode["input_events"] for episode in episodes)
            spikes = sum(episode["spikes"] for episode in episodes)
            conditions[name] = {
                "quartile_mean_steps": [
                    sum(episode["steps"] for episode in episodes[start:start + 32]) / 32
                    for start in (0, 32, 64, 96)
                ],
                "episodes_at_least_250_steps": sum(
                    episode["steps"] >= 250 for episode in episodes
                ),
                "spike_to_input_ratio": spikes / inputs,
                "terminal_updates": sum(
                    episode["terminal_updates"] for episode in episodes
                ),
            }
        summaries.append({"run_id": run["run_id"], "conditions": conditions})
    return summaries


def refractory_gate_probe(present: Sequence[bool]) -> dict:
    """Compare the configured neuron with a two-step input-suppression gate."""
    neuron = Neuron(0, num_branches=1)
    cooldown = 0
    observed = []
    expected = []
    membrane = []
    for active in present:
        if type(active) is not bool:
            raise ValueError("Probe events must be boolean")
        if active:
            neuron.add_input_to_branch(0, 1.6)
        observed.append(neuron.step())
        if cooldown:
            expected.append(False)
            cooldown -= 1
        elif active:
            expected.append(True)
            cooldown = 2
        else:
            expected.append(False)
        membrane.append(neuron.v)
    return {"observed": tuple(observed), "refractory_mask": tuple(expected),
            "membrane": tuple(membrane)}
