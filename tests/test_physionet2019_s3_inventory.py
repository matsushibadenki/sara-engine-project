"""PhysioNet inventory must use object metadata only and reject unexpected paths."""
import importlib.util
from hashlib import sha256
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval/physionet2019_s3_inventory.py"
SPEC = importlib.util.spec_from_file_location("physionet2019_s3_inventory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
summarize_objects = MODULE.summarize_objects


PROTOCOL = {"prefix": "challenge-2019/1.0.0/training/",
            "expected_sets": {"training_setA": 1, "training_setB": 1},
            "allowed_object_suffix": ".psv"}


def entry(set_name, patient, size=10):
    return {"Key": f"{PROTOCOL['prefix']}{set_name}/{patient}.psv",
            "Size": size, "ETag": '"deadbeef"'}


def test_exact_inventory_passes_without_object_body():
    result = summarize_objects([{"Contents": [entry("training_setA", "p000001"),
                                               entry("training_setB", "p000002")]}], PROTOCOL)
    assert result["passed"]
    assert result["total_bytes"] == {"training_setA": 10, "training_setB": 10}
    assert result["object_bodies_read"] is False


@pytest.mark.parametrize("bad", [
    entry("training_setC", "p000001"),
    entry("training_setA", "../p000001"),
    entry("training_setA", "patient001"),
    {"Key": "other/training_setA/p000001.psv", "Size": 10, "ETag": '"x"'},
])
def test_unexpected_paths_fail_closed(bad):
    with pytest.raises(ValueError, match="Unexpected"):
        summarize_objects([{"Contents": [bad]}], PROTOCOL)


def test_duplicate_or_zero_size_fails_closed():
    good = entry("training_setA", "p000001")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_objects([{"Contents": [good, good]}], PROTOCOL)
    with pytest.raises(ValueError, match="metadata"):
        summarize_objects([{"Contents": [entry("training_setA", "p000001", 0)]}], PROTOCOL)


def test_official_inventory_result_is_pinned_and_metadata_only():
    root = Path(__file__).resolve().parents[1]
    protocol = root / "data/processed/benchmark_fixtures/physionet2019_s3_inventory_protocol_v1.json"
    result_path = root / "workspace/evaluation/physionet2019_s3_inventory_v1.json"
    assert sha256(protocol.read_bytes()).hexdigest() == (
        "203471464783a4573e6d7671d696418d941ac6404cf7ad730a6a83b6bbfbc6a1")
    assert sha256(result_path.read_bytes()).hexdigest() == (
        "7aea0182cceaa0051b76fd6dad74521fb7e3ff5e8e840e16f664bce4533141d2")
    result = json.loads(result_path.read_text())
    assert result["passed"] is True
    assert result["object_counts"] == {"training_setA": 20336, "training_setB": 20000}
    assert sum(result["total_bytes"].values()) == 255057484
    assert result["object_bodies_read"] is False
    assert result["patient_labels_read"] is False
    assert result["descriptive_values_read"] is False
    assert result["candidate_scored"] is False
