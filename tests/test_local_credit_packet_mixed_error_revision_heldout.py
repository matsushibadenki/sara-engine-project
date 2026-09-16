import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data/processed/benchmark_fixtures"
EVALUATION = ROOT / "workspace/evaluation"
PROTOCOL = FIXTURES / "local_credit_packet_mixed_error_revision_heldout_v1.json"
EXECUTION = FIXTURES / "local_credit_packet_mixed_error_revision_heldout_execution_v1.json"
ROWS = FIXTURES / "local_credit_packet_mixed_error_revision_heldout_rows_v1.jsonl"
AUDIT = EVALUATION / "local_credit_packet_mixed_error_revision_heldout_audit_v1.json"
RESULT = EVALUATION / "local_credit_packet_mixed_error_revision_heldout_result_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mixed_error_heldout_inputs_and_independent_audit_are_bound():
    protocol = json.loads(PROTOCOL.read_text())
    execution = json.loads(EXECUTION.read_text())
    audit = json.loads(AUDIT.read_text())
    assert _sha(PROTOCOL) == execution["protocol_sha256"] == audit["protocol_sha256"]
    assert _sha(ROWS) == execution["rows_sha256"] == audit["rows_sha256"]
    assert _sha(AUDIT) == execution["pre_execution_audit_sha256"]
    assert _sha(ROOT / "scripts/eval/local_credit_packet_mixed_error_revision.py") == execution["candidate_source_sha256"]
    assert _sha(ROOT / "scripts/eval/local_credit_packet_mixed_error_revision_heldout_materialize.py") == execution["materializer_source_sha256"]
    assert _sha(EVALUATION / "local_credit_packet_mixed_error_revision_development_v1.json") == protocol["parent_development_result_sha256"]
    assert audit["passed"] is True
    assert all(audit["checks"].values())
    assert audit["direct_shortcut_probe"] == 0.5
    assert audit["row_count"] == 3200
    oracle = audit["independent_oracle"]
    assert oracle["passed"] is True
    assert all(all(group.values()) for group in oracle["checks"].values())
    assert all(value["conditions"]["full"]["global_accuracy"] == 1.0
               for value in oracle["per_seed"].values())
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    assert len(rows) == 3200
    for seed in protocol["identity"]["seeds"]:
        assert Counter(row["phase"] for row in rows if row["seed"] == seed) == Counter({
            "training": 512, "test": 128,
        })


def test_mixed_error_heldout_negative_gate_is_preserved():
    result = json.loads(RESULT.read_text())
    assert result["heldout_consumed"] is True
    assert result["heldout_gate_passed"] is False
    assert result["harness_passed"] is True
    assert result["heldout_checks"]["five_seed_simple_gains"] is False
    assert all(value for key, value in result["heldout_checks"].items()
               if key != "five_seed_simple_gains")
    assert all(result["harness_checks"].values())
    assert result["accuracy_by_condition"]["full"] == {
        "prior_credit_only": 0.5,
        "simple_replay": 0.9,
        "correction_packet": 1.0,
        "direct_outcome_correction": 0.425,
        "oracle_control": 1.0,
    }
    assert result["accuracy_by_condition"]["partial_six"]["correction_packet"] == 0.8
    assert result["accuracy_by_condition"]["one_corrupt"]["correction_packet"] == 0.78125
    assert sum(seed["conditions"]["full"]["arms"]["simple_replay"]["accuracy"] == 1.0
               for seed in result["per_seed"].values()) == 2
    assert max(row["peak_state_bytes"] for seed in result["per_seed"].values()
               for condition in seed["conditions"].values()
               for row in list(condition["arms"].values()) + list(condition["controls"].values())) == 7621
    assert result["production_authorized"] is False
