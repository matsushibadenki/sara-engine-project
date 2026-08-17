from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "continual_horizon_collection_request.py"
    spec = importlib.util.spec_from_file_location("continual_horizon_collection_request", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collection_request_lists_blocked_horizon_targets():
    module = _load_module()
    gate = {
        "schema": "test-gate",
        "promotion_allowed": False,
        "promotion_checks": {
            "independent_manifest_quality": True,
            "required_horizon_buckets_present": False,
        },
        "source_domains": ["one.example", "two.example"],
    }
    targets = module.build_targets(gate)

    assert targets["target_count"] == 2
    assert targets["blocked_promotion_checks"] == ["required_horizon_buckets_present"]
    assert targets["targets"][0]["required_horizon_buckets"] == [10, 30, 100]
    assert "near_duplicate_signature" in targets["targets"][0]["required_fields"]


def test_collection_request_is_empty_after_promotion():
    module = _load_module()
    targets = module.build_targets(
        {
            "promotion_allowed": True,
            "promotion_checks": {
                "independent_manifest_quality": True,
                "required_horizon_buckets_present": True,
            },
            "source_domains": ["one.example"],
        }
    )

    assert targets["target_count"] == 0
    assert targets["targets"] == []


def test_collection_request_omits_already_observed_horizon_buckets():
    module = _load_module()
    targets = module.build_targets(
        {
            "promotion_allowed": False,
            "promotion_checks": {"required_horizon_buckets_present": False},
            "source_domains": ["one.example", "two.example"],
            "domain_horizons": {
                "one.example": list(range(11)),
                "two.example": list(range(31)),
            },
        }
    )

    assert targets["targets"][0]["required_horizon_buckets"] == [30, 100]
    assert targets["targets"][1]["required_horizon_buckets"] == [100]
