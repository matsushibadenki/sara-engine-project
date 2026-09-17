"""Fixed development smoke runner exercises the durable capture path."""
from hashlib import sha256
import importlib.util
from pathlib import Path
import tempfile

import pytest

from sara_engine.evaluation.event_unit_causal_isolation import generate_episodes
from sara_engine.research import CaptureOutcome, DiscoveryCaptureLog
from sara_engine.utils.project_paths import ensure_output_directory, workspace_path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval/discovery_capture_event_unit_smoke.py"


def _module():
    spec = importlib.util.spec_from_file_location("discovery_capture_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pilot_captures_one_development_action_without_approval():
    module = _module()
    parent = ensure_output_directory(workspace_path("research_capture_tests"))
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        first_path = str(Path(temporary) / "first.jsonl")
        second_path = str(Path(temporary) / "second.jsonl")
        first = module.run_pilot(first_path)
        second = module.run_pilot(second_path)
        view = DiscoveryCaptureLog(first_path).load()
        assert len(view.events) == 3
        assert view.pending_node_ids == ()
        assert isinstance(view.events[-1], CaptureOutcome)
        assert view.events[-1].score == first["accuracy"]
        assert first["capture_head_sha256"] == view.head_sha256
        assert first["promotion_authorized"] is False
        assert first["accuracy"] == second["accuracy"]
        assert first["prediction_trace_sha256"] == second["prediction_trace_sha256"]
        assert first["state_bytes"] == second["state_bytes"]
        assert first["candidate_sha256"] == sha256(Path(module.CANDIDATE_PATH).read_bytes()).hexdigest()
        assert first["protocol_sha256"] == sha256(Path(module.PROTOCOL_PATH).read_bytes()).hexdigest()
        episodes = [
            *generate_episodes(seeds=(module.SEED,), count_per_family=4,
                               split="training", namespace="capture-smoke-v1"),
            *generate_episodes(seeds=(module.SEED,), count_per_family=2,
                               split="development", namespace="capture-smoke-v1"),
        ]
        assert first["input_event_count"] == sum(len(row.events) for row in episodes)
        before = Path(first_path).read_bytes()
        with pytest.raises(FileExistsError):
            module.run_pilot(first_path)
        assert Path(first_path).read_bytes() == before
