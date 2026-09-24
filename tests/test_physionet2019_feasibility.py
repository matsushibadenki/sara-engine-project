"""PhysioNet feasibility reads only fixed label/time metadata semantics."""
from hashlib import sha256
import json
from pathlib import Path
import importlib.util

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval/physionet2019_feasibility.py"
SPEC = importlib.util.spec_from_file_location("physionet2019_feasibility", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_patient(path, rows):
    path.write_text("Feature|ICULOS|SepsisLabel\n" + "\n".join(
        f"secret-{hour}|{hour}|{label}" for hour, label in rows) + "\n")


def test_patient_summary_selects_fixed_metadata_columns(tmp_path):
    path = tmp_path / "patient.psv"
    write_patient(path, [(1, 0), (2, 1)])
    assert MODULE.patient_summary(path, ["ICULOS", "SepsisLabel"]) == ([1, 2], [0, 1])


def test_sequence_validation_rejects_gaps_and_label_reversal():
    with pytest.raises(ValueError, match="continuous"):
        MODULE.validate_sequence([1, 3], [0, 0])
    with pytest.raises(ValueError, match="non-decreasing"):
        MODULE.validate_sequence([1, 2], [1, 0])
    with pytest.raises(ValueError, match="binary"):
        MODULE.validate_sequence([1], [2])


def test_split_is_deterministic_and_set_b_is_heldout():
    protocol = {"hash_namespace": "fixed", "split": {"training_setA": {
        "development_remainders": [4]}}}
    assert MODULE.assigned_split("training_setB", "training_setB/p.psv", protocol) == "heldout"
    first = MODULE.assigned_split("training_setA", "training_setA/p.psv", protocol)
    assert first in {"training", "development"}
    assert MODULE.assigned_split("training_setA", "training_setA/p.psv", protocol) == first


def test_non_one_based_sequence_is_ineligible():
    with pytest.raises(ValueError, match="continuous one-based"):
        MODULE.validate_sequence([2, 3, 4], [0, 0, 0])


def test_negative_feasibility_result_is_pinned_and_feature_free():
    protocol = ROOT / "data/processed/benchmark_fixtures/physionet2019_feasibility_protocol_v1.json"
    result_path = ROOT / "workspace/evaluation/physionet2019_feasibility_v1.json"
    assert sha256(protocol.read_bytes()).hexdigest() == MODULE.PROTOCOL_SHA256
    assert sha256(result_path.read_bytes()).hexdigest() == (
        "31a985c16bf688eb7d8e0c715695bdde2a020914d7a783a2381683502abaee11")
    result = json.loads(result_path.read_text())
    assert result["passed"] is False
    assert [result["case_counts"][split]["positive"]
            for split in ("training", "development", "heldout")] == [107, 29, 78]
    assert result["all_other_columns_interpreted_or_stored"] is False
    assert result["clinical_features_materialized"] is False
    assert result["candidate_scored"] is False
    assert result["preprotocol_inspected_patient_excluded"] is True
