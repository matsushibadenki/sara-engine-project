import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data" / "processed" / "benchmark_fixtures" / "local_credit_packet_multihop_heldout_v1.json"
ROWS = ROOT / "data" / "processed" / "benchmark_fixtures" / "local_credit_packet_multihop_heldout_rows_v1.jsonl"
MATERIALIZATION = ROOT / "workspace" / "evaluation" / "local_credit_packet_multihop_heldout_materialization.json"
RESULT = ROOT / "workspace" / "evaluation" / "local_credit_packet_multihop_heldout_result_v1.json"
RESULT_SHA256 = "6c4beca8b9917485ef1d532c0483964b1caf837e7c964fa9c63fc1368cf2ec87"


def test_one_shot_result_is_pinned_and_heldout_was_consumed_once():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == RESULT_SHA256
    protocol = json.loads(PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    assert result["protocol_sha256"] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert result["rows_sha256"] == hashlib.sha256(ROWS.read_bytes()).hexdigest()
    assert result["execution_count"] == 1
    assert result["heldout_consumed"] is True
    assert result["production_authorized"] is False
    for name in ("candidate", "packet"):
        path = ROOT / protocol["frozen_sources"][f"{name}_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == result["source_sha256"][name]


def test_independent_oracle_and_all_five_seed_gates_pass():
    materialization = json.loads(MATERIALIZATION.read_text())
    result = json.loads(RESULT.read_text())
    assert materialization["checks"]["independent_oracle_agreement"] is True
    assert result["harness_passed"] is True
    assert result["heldout_gate_passed"] is True
    assert len(result["per_seed"]) == 5
    assert all(
        seed_result["arms"]["local_credit_packet"]["accuracy"] == 1.0
        for seed_result in result["per_seed"].values()
    )
    assert result["arms"]["direct_anchor_shortcut"]["accuracy"] == 0.5
    assert result["controls"]["B_local_map_reset"]["accuracy"] == 0.5
