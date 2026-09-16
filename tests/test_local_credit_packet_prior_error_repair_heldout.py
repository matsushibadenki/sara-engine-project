import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data/processed/benchmark_fixtures"
EVALUATION = ROOT / "workspace/evaluation"
PROTOCOL = FIXTURES / "local_credit_packet_prior_error_repair_heldout_v1.json"
EXECUTION = FIXTURES / "local_credit_packet_prior_error_repair_heldout_execution_v1.json"
ROWS = FIXTURES / "local_credit_packet_prior_error_repair_heldout_rows_v1.jsonl"
AUDIT = EVALUATION / "local_credit_packet_prior_error_repair_heldout_audit_v1.json"
RESULT = EVALUATION / "local_credit_packet_prior_error_repair_heldout_result_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repair_heldout_inputs_and_independent_oracle_are_bound():
    protocol = json.loads(PROTOCOL.read_text())
    execution = json.loads(EXECUTION.read_text())
    audit = json.loads(AUDIT.read_text())
    assert _sha(PROTOCOL) == execution["protocol_sha256"]
    assert _sha(ROWS) == execution["rows_sha256"] == audit["rows_sha256"]
    assert _sha(AUDIT) == execution["pre_execution_audit_sha256"]
    assert _sha(ROOT / "scripts/eval/local_credit_packet_prior_error_repair.py") == execution["candidate_source_sha256"]
    assert _sha(ROOT / "scripts/eval/local_credit_packet_prior_error_repair_heldout_materialize.py") == execution["materializer_source_sha256"]
    assert _sha(EVALUATION / "local_credit_packet_prior_error_repair_development_v1.json") == protocol["parent_development_result_sha256"]
    assert audit["passed"] is True
    assert all(audit["checks"].values())
    assert audit["row_count"] == 3200
    assert audit["direct_shortcut_probe"] == 0.5
    assert audit["repair_oracle"]["opposite_sign_count"] == 40
    assert audit["repair_oracle"]["wrong_prediction_accuracy"] == 0.0
    assert audit["repair_oracle"]["corrected_prediction_accuracy"] == 1.0
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    assert len(rows) == 3200
    for seed in protocol["identity"]["seeds"]:
        assert Counter(row["phase"] for row in rows if row["seed"] == seed) == Counter({
            "training": 512, "test": 128,
        })


def test_repair_heldout_one_shot_gate_and_packet_contract():
    result = json.loads(RESULT.read_text())
    assert result["heldout_consumed"] is True
    assert result["heldout_gate_passed"] is True
    assert result["harness_passed"] is True
    assert result["mean_accuracy"] == {
        "wrong_credit_only": 0.0,
        "simple_replay": 0.5,
        "correction_packet": 1.0,
        "direct_outcome_correction": 0.525,
        "oracle_control": 1.0,
    }
    assert result["control_accuracy"]["magnitude_capped_at_one"] == 0.5
    assert result["control_accuracy"]["replay_disabled"] == 0.0
    assert result["control_accuracy"]["A_anchor_erased_before_replay"] == 0.0
    assert result["control_accuracy"]["B_to_A_route_shuffle_on_replay"] == 0.0
    assert result["control_accuracy"]["B_to_A_sign_shuffle_on_replay"] == 0.0
    for seed in result["per_seed"].values():
        corrected = seed["arms"]["correction_packet"]
        assert corrected["accuracy"] == 1.0
        assert corrected["pre_revision_A_accuracy"] == 0.0
        assert corrected["early_packets"] == 8
        assert corrected["replay_packets"] == corrected["replay_lookups"] == 8
        assert corrected["prior_sign_reversals"] == 8
        assert corrected["magnitude_two_packets"] == 8
        assert corrected["maximum_A_anchors"] == corrected["maximum_B_anchors"] == 8
        assert corrected["maximum_packet_bytes"] == 21
        assert corrected["B_local_updates"] == 16
        assert corrected["B_local_accuracy"] == 1.0
    assert result["production_authorized"] is False
