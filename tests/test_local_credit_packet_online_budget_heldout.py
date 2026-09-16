import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data/processed/benchmark_fixtures"
EVALUATION = ROOT / "workspace/evaluation"
PROTOCOL = FIXTURES / "local_credit_packet_online_budget_heldout_v1.json"
EXECUTION = FIXTURES / "local_credit_packet_online_budget_heldout_execution_v1.json"
ROWS = FIXTURES / "local_credit_packet_online_budget_heldout_rows_v1.jsonl"
AUDIT = EVALUATION / "local_credit_packet_online_budget_heldout_audit_v1.json"
RESULT = EVALUATION / "local_credit_packet_online_budget_heldout_result_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_independent_heldout_inputs_and_audit_are_bound():
    protocol = json.loads(PROTOCOL.read_text())
    execution = json.loads(EXECUTION.read_text())
    audit = json.loads(AUDIT.read_text())
    assert _sha(PROTOCOL) == execution["protocol_sha256"]
    assert _sha(ROWS) == execution["rows_sha256"] == audit["rows_sha256"]
    assert _sha(AUDIT) == execution["pre_execution_audit_sha256"]
    assert _sha(ROOT / "scripts/eval/local_credit_packet_online_budget_curve.py") == execution["candidate_source_sha256"]
    assert _sha(EVALUATION / "local_credit_packet_online_budget_curve_development_v1.json") == protocol["parent_development_result_sha256"]
    assert audit["passed"] is True
    assert all(audit["checks"].values())
    assert audit["row_count"] == 3200
    assert audit["direct_shortcut_probe"] == 0.5
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    assert len(rows) == 3200
    for seed in protocol["identity"]["seeds"]:
        seed_rows = [row for row in rows if row["seed"] == seed]
        assert Counter(row["phase"] for row in seed_rows) == Counter({"training": 512, "test": 128})


def test_one_shot_heldout_gate_and_resource_contract():
    result = json.loads(RESULT.read_text())
    assert result["heldout_consumed"] is True
    assert result["heldout_gate_passed"] is True
    assert result["harness_passed"] is True
    assert result["accuracy_by_online_budget"] == {
        "0": {
            "no_A_credit": 0.5, "online_only_packet": 0.5,
            "targeted_replay_packet": 1.0,
            "targeted_replay_direct_shortcut": 0.525,
            "oracle_control": 1.0,
        },
        "2": {
            "no_A_credit": 0.5, "online_only_packet": 0.575,
            "targeted_replay_packet": 1.0,
            "targeted_replay_direct_shortcut": 0.5,
            "oracle_control": 1.0,
        },
    }
    assert result["zero_budget_control_accuracy"]["replay_disabled"] == 0.5
    assert result["zero_budget_control_accuracy"]["A_anchor_erased_before_replay"] == 0.5
    assert result["zero_budget_control_accuracy"]["B_to_A_route_shuffle_on_replay"] == 0.0
    assert result["zero_budget_control_accuracy"]["B_to_A_sign_shuffle_on_replay"] == 0.0
    for seed in result["per_seed"].values():
        for budget in ("0", "2"):
            replay = seed["budgets"][budget]["arms"]["targeted_replay_packet"]
            assert replay["accuracy"] == 1.0
            assert replay["retained_sources"] == 8
            assert replay["replay_packets"] == replay["replay_lookups"] == 8
            assert replay["maximum_A_anchors"] == replay["maximum_B_anchors"] == 8
            assert replay["B_local_updates"] == 8
            assert replay["B_local_accuracy"] == 1.0
            assert replay["maximum_packet_bytes"] == 21
    assert result["production_authorized"] is False
