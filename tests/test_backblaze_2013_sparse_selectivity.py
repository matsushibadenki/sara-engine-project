"""Frozen training-only Backblaze sparse-selectivity diagnostic."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/backblaze_2013_sparse_selectivity_protocol_v1.json"
RESULT = ROOT / "workspace/evaluation/backblaze_2013_sparse_selectivity_diagnostic_v1.json"


def test_sparse_selectivity_protocol_and_result_are_frozen():
    assert sha256(PROTOCOL.read_bytes()).hexdigest() == (
        "b39992eec7ad5e7e2e5127680db90ad0d0acb065eb48c6d3a11c2842b924a15d")
    assert sha256(RESULT.read_bytes()).hexdigest() == (
        "794ef9f44cd95d9ba0247eae49cb04319fd767f9e8276256b053adeb7be49464")


def test_signed_delta_gain_is_retained_without_promoting_failed_gate():
    result = json.loads(RESULT.read_text())
    signed = result["encodings"]["sparse_signed_log_delta"]
    baseline = result["encodings"]["v1_presence_onset_increase"]
    assert signed["balanced_accuracy"] > baseline["balanced_accuracy"] + 0.17
    assert min(signed["fold_balanced_accuracy"]) > 0.69
    assert signed["event_cap_fraction"] == 0
    assert result["checks"]["route_selectivity"] is False
    assert result["diagnostic_passed"] is False
    assert result["candidate_promotion_authorized"] is False
    assert result["development_consumed"] is False
    assert result["heldout_consumed"] is False
