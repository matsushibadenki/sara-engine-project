"""V2 protocol/preflight tests do not score real CartPole candidates."""
from hashlib import sha256
import json
from pathlib import Path

import pytest

from sara_engine.evaluation import cartpole_registered_runner_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/cartpole_terminal_prereg_v2.json"
TASK = ROOT / "data/processed/benchmark_fixtures/cartpole_terminal_task_v2.json"
V1_TASK = ROOT / "data/processed/benchmark_fixtures/cartpole_terminal_task_v1.json"


def _digest():
    return sha256(PROTOCOL.read_bytes()).hexdigest()


def test_v2_protocol_pins_unchanged_gate_and_new_task_identity():
    protocol = v2.load_frozen_protocol_v2(_digest())
    parent = v2.v1.load_frozen_protocol(v2.V1_PROTOCOL_SHA256)
    assert protocol["gate"] == parent["gate"]
    assert protocol["resource_ceiling"] == parent["resource_ceiling"]
    assert protocol["variants"] == parent["variants"]
    assert protocol["heldout_opening_authorized"] is False
    assert protocol["task_sha256"] == sha256(TASK.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="digest mismatch"):
        v2.load_frozen_protocol_v2("0" * 64)


def test_v2_seed_ranges_are_disjoint_from_all_v1_ranges():
    old = json.loads(V1_TASK.read_text())["split_plan"]
    new = json.loads(TASK.read_text())
    old_seeds = {
        old[split]["split_base"] + 1000 * run_id + index
        for split in ("training", "development", "heldout")
        for run_id in old["run_ids"]
        for index in range(old[split]["episodes_per_run"])
    }
    new_seeds = {
        new["split_bases"][split] + 1000 * run_id + index
        for split in ("training", "development", "heldout")
        for run_id in new["run_ids"]
        for index in range(new["episodes_per_run"][split])
    }
    assert not old_seeds.intersection(new_seeds)
    assert len(new_seeds) == 5 * (128 + 32 + 32)
    assert new["heldout_sealed"] is True


def test_serialization_preflight_fails_before_one_shot_reservation(monkeypatch, tmp_path):
    output = tmp_path / "development.json"
    monkeypatch.setattr(v2, "ensure_allowed_output_path", lambda _: str(output))
    monkeypatch.setattr(v2, "inspect_environment", lambda: {"action_count": object()})
    with pytest.raises(ValueError, match="Unexpected environment audit fields"):
        v2.run_registered_development_v2(expected_protocol_sha256=_digest())
    assert not output.exists()
    assert not Path(str(output) + ".lock").exists()
