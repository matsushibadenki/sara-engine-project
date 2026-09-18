"""Unscored independent held-out materialization invariants."""
from copy import deepcopy
import json
import runpy
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1] / "scripts/eval"
          / "local_credit_packet_nonoracle_anchor_heldout_materialize.py")


def _fixture():
    module = runpy.run_path(str(SCRIPT))
    protocol = json.loads(module["PROTOCOL"].read_text())
    return module, protocol, module["generate"](protocol)


def test_independent_materialization_has_balanced_new_identities():
    module, protocol, rows = _fixture()
    audit = module["independent_audit"](rows, protocol)
    assert audit["passed"]
    assert audit["rows"] == 5 * (512 + 128)
    assert audit["changed_sign_strata"]["2"] == {"0": 8, "1": 8}
    assert all(audit["checks"].values())
    assert not audit["candidate_scored"]
    parent = json.loads(module["PARENT_PROTOCOL"].read_text())
    assert set(protocol["identity"]["seeds"]).isdisjoint(parent["identity"]["seeds"])
    assert set(protocol["identity"]["A_cues"]).isdisjoint(parent["identity"]["A_cues"])
    assert set(protocol["identity"]["B_cues"]).isdisjoint(parent["identity"]["B_cues"])


def test_independent_audit_rejects_corrupted_outcome_and_order():
    module, protocol, rows = _fixture()
    corrupt = deepcopy(rows)
    corrupt[0]["success"] ^= 1
    assert not module["independent_audit"](corrupt, protocol)["passed"]
    reordered = deepcopy(rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert not module["independent_audit"](reordered, protocol)["passed"]


def test_anchor_selection_does_not_consult_success():
    module, protocol, rows = _fixture()
    initial = module["independent_audit"](rows, protocol)["first_two_identities"]
    tampered = deepcopy(rows)
    for row in tampered:
        if row["phase"] == "training":
            row["success"] ^= 1
    after = module["independent_audit"](tampered, protocol)["first_two_identities"]
    assert after == initial
