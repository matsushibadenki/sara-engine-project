import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/benchmark_fixtures/local_credit_packet_mixed_error_replay_parity_diagnostic_v1.json"
REPORT = ROOT / "workspace/evaluation/local_credit_packet_mixed_error_replay_parity_diagnostic_v1.json"


def test_replay_parity_uses_only_pinned_development_inputs():
    protocol = json.loads(PROTOCOL.read_text())
    report = json.loads(REPORT.read_text())
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == report["protocol_sha256"]
    assert report["heldout_inputs_used"] is False
    assert report["exploratory_only"] is True
    assert report["passed"] is True
    assert all(report["consistency_checks"].values())
    for path, key in (
        (ROOT / "data/processed/benchmark_fixtures/local_credit_packet_mixed_error_revision_v1.json", "development_protocol_sha256"),
        (ROOT / "workspace/evaluation/local_credit_packet_mixed_error_revision_development_v1.json", "development_result_sha256"),
        (ROOT / "scripts/eval/local_credit_packet_noisy_late_B_evidence_v2.py", "selector_source_sha256"),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol[key]


def test_replay_parity_tie_mechanism():
    report = json.loads(REPORT.read_text())
    assert report["summary"] == {
        "cue_count": 40,
        "changed_sign_histogram": {"0": 9, "1": 21, "2": 10},
        "simple_correct_by_changed_count": {"0": 9, "1": 21, "2": 5},
        "second_step_necessary_by_changed_count": {"0": 0, "1": 0, "2": 5},
        "simple_tie_correct_count": 5,
        "simple_tie_wrong_count": 5,
    }
    changed_twice = [cue for seed in report["per_seed"].values()
                     for cue in seed["per_cue"].values()
                     if cue["changed_sign_count"] == 2]
    assert len(changed_twice) == 10
    assert all(cue["simple_margin"] == 0 and cue["simple_tie"]
               for cue in changed_twice)
    assert all(cue["second_step_necessary"] == (cue["target"] == 1)
               for cue in changed_twice)
    assert all(cue["correction_correct"] for cue in changed_twice)
