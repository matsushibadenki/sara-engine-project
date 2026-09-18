"""CartPole task interface checks do not score a candidate policy."""
import math
import platform
from hashlib import sha256
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.cartpole_sparse_reward import (
    encode_observation, inspect_environment, terminal_feedback,
)


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "data/processed/benchmark_fixtures/cartpole_terminal_task_v1.json"


def test_task_contract_pins_interface_and_separates_all_episode_seeds():
    contract = json.loads(TASK.read_text(encoding="utf-8"))
    interface = contract["interface"]
    assert sha256((ROOT / interface["source_path"]).read_bytes()).hexdigest() == interface["source_sha256"]
    plan = contract["split_plan"]
    all_seeds = []
    for name in ("training", "development", "heldout"):
        section = plan[name]
        for run_id in plan["run_ids"]:
            all_seeds.extend(section["split_base"] + 1000 * run_id + index
                             for index in range(section["episodes_per_run"]))
    assert len(all_seeds) == 5 * (128 + 32 + 32)
    assert len(set(all_seeds)) == len(all_seeds)
    assert plan["heldout"]["sealed"] is True
    assert contract["boundaries"]["candidate_scoring_authorized"] is False
    assert contract["boundaries"]["frozen_heldout_opened"] is False
    assert contract["environment"]["default_step_reward_used_for_learning"] is False


def test_event_encoder_is_bounded_and_keeps_time_explicit():
    first = encode_observation((0.0, 0.0, 0.0, 0.0), 0)
    later = encode_observation((-2.0, -2.0, -0.2, 2.0), 7)
    assert [event.route for event in first] == [2, 7, 12, 17]
    assert [event.route for event in later] == [0, 5, 10, 19]
    assert all(event.time == 7 and event.branch == 0 for event in later)
    assert all(0 <= event.route < 20 for event in (*first, *later))
    with pytest.raises(ValueError):
        encode_observation((0.0, 0.0, 0.0), 0)
    with pytest.raises(ValueError):
        encode_observation((0.0, 0.0, math.inf, 0.0), 0)
    with pytest.raises(ValueError):
        encode_observation((0.0, 0.0, 0.0, 0.0), 500)


def test_sparse_feedback_exists_only_at_episode_end():
    assert terminal_feedback(steps=1, terminated=True, truncated=False) == 1 / 500
    assert terminal_feedback(steps=500, terminated=False, truncated=True) == 1.0
    with pytest.raises(ValueError):
        terminal_feedback(steps=20, terminated=False, truncated=False)
    with pytest.raises(ValueError):
        terminal_feedback(steps=0, terminated=True, truncated=False)


def test_installed_standard_environment_matches_task_contract():
    gym = pytest.importorskip("gymnasium")
    numpy = pytest.importorskip("numpy")
    if (gym.__version__, platform.python_version(), numpy.__version__) != (
        "1.0.0", "3.10.18", "1.25.2"
    ):
        with pytest.raises(ValueError, match="runtime differs"):
            inspect_environment()
        return
    audit = inspect_environment()
    assert audit == {
        "environment_id": "CartPole-v1",
        "gymnasium_version": "1.0.0",
        "python_version": "3.10.18",
        "numpy_version": "1.25.2",
        "max_episode_steps": 500,
        "observation_channels": 4,
        "events_per_observation": 4,
        "action_count": 2,
        "evaluator_invoked": False,
    }
