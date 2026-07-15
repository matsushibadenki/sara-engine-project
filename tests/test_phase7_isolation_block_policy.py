import importlib.util
import os
import sys


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "phase7_isolation_block_policy.py"))
    spec = importlib.util.spec_from_file_location("phase7_isolation_block_policy", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _targets():
    return {"targets": [{"request_id": "fixture_gap"}, {"request_id": "normal_gap"}]}


def test_phase7_isolation_block_policy_blocks_fixture_requests_on_overlap():
    module = _load_module()
    policy = module.build_policy({"passed": False, "checks": {"source_hash_isolated": False, "source_domain_isolated": False}}, _targets())
    assert policy["action"] == "blocked"
    assert policy["targets"]["blocked_request_ids"] == ["fixture_gap"]
    assert policy["targets"]["blocked_request_missing_axes"]["fixture_gap"] == ["source_domain_isolated", "source_hash_isolated"]


def test_phase7_isolation_block_policy_releases_only_its_own_axes():
    module = _load_module()
    targets = _targets()
    targets["blocked_request_ids"] = ["fixture_gap", "normal_gap"]
    targets["blocked_request_missing_axes"] = {"fixture_gap": ["source_hash_isolated"], "normal_gap": ["manual_review"]}
    policy = module.build_policy({"passed": True, "checks": {}}, targets)
    assert policy["action"] == "released"
    assert policy["targets"]["blocked_request_ids"] == ["normal_gap"]
    assert "fixture_gap" not in policy["targets"]["blocked_request_missing_axes"]


def test_phase7_isolation_block_policy_blocks_invalid_signature_format():
    module = _load_module()
    policy = module.build_policy(
        {
            "passed": False,
            "checks": {"near_duplicate_signature_format_valid": False},
        },
        {"targets": [{"request_id": "fixture_gap"}]},
    )
    assert policy["failed_axes"] == ["near_duplicate_signature_format_valid"]
    assert policy["targets"]["blocked_request_missing_axes"]["fixture_gap"] == [
        "near_duplicate_signature_format_valid"
    ]
