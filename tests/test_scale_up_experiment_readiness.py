from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "scale_up_experiment_readiness.py"
    spec = importlib.util.spec_from_file_location("scale_up_experiment_readiness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scale_up_readiness_stays_blocked_until_gates_pass():
    module = _load_module()
    report = module.build_readiness(
        {"promotion_allowed": False},
        {"promotion_allowed": False},
        {"promotion_allowed": False},
    )

    assert report["ready_to_execute"] is False
    assert report["plan"]["episode_buckets"] == [1000, 10000]
    assert "next_level_promotion_gate_blocked" in report["blockers"]


def test_scale_up_readiness_can_open_only_when_both_gates_pass():
    module = _load_module()
    report = module.build_readiness(
        {"promotion_allowed": True},
        {"promotion_allowed": True},
        {"promotion_allowed": True},
    )

    assert report["ready_to_execute"] is True
    assert report["plan"]["execution_policy"]["external_device_required"] is False
