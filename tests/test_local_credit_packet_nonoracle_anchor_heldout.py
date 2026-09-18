"""Unscored identity and gate tests for the one-shot held-out scorer."""
import json
import runpy
from pathlib import Path
from hashlib import sha256

import pytest


SCRIPT = (Path(__file__).resolve().parents[1] / "scripts/eval"
          / "local_credit_packet_nonoracle_anchor_heldout.py")


def _module_and_protocol():
    module = runpy.run_path(str(SCRIPT))
    return module, json.loads(module["PROTOCOL"].read_text())


def test_frozen_result_refuses_reopening():
    module, _ = _module_and_protocol()
    assert module["OUTPUT"].exists()
    with pytest.raises(ValueError, match="already reserved or completed"):
        module["verify_inputs"]()


def test_frozen_result_matches_registered_gate():
    module, _ = _module_and_protocol()
    output = module["OUTPUT"]
    assert sha256(output.read_bytes()).hexdigest() == (
        "802ae42b5dc93b912478f80f91fc0db524796684b933596c90c0ed44f3c81e70")
    result = json.loads(output.read_text())
    assert result["heldout_gate_passed"] is True
    assert all(result["checks"].values())
    assert all(result["oracle_checks"].values())
    assert all(result["resource_checks"].values())
    assert result["selective_replay_packets"] == 78
    assert result["unconditional_replay_packets"] == 160
    assert result["test_updates"] is False


def test_identity_preflight_rejects_changed_rows(tmp_path):
    module, _ = _module_and_protocol()
    modified = tmp_path / "rows.jsonl"
    modified.write_bytes(module["ROWS"].read_bytes() + b"\n")
    module["verify_inputs"].__globals__["ROWS"] = modified
    module["verify_inputs"].__globals__["OUTPUT"] = tmp_path / "not_scored.json"
    with pytest.raises(ValueError, match="Pinned held-out source or data changed"):
        module["verify_inputs"]()


def test_gate_requires_packet_saving_and_exact_oracle():
    module, protocol = _module_and_protocol()
    per_seed = {}
    for seed in protocol["identity"]["seeds"]:
        arms = {}
        for arm in protocol["evaluation"]["arms"]:
            per_cue = {}
            for cue, (target, changed) in enumerate(((0, 2), (1, 2), (0, 0), (1, 0))):
                correct = arm in ("selective_two_step", "unconditional_two_step")
                if arm == "simple_tie_zero":
                    correct = target == 0
                elif arm == "simple_tie_one":
                    correct = target == 1
                per_cue[str(cue)] = {"target": target, "changed_sign_count": changed,
                                     "correct": correct}
            arms[arm] = {
                "accuracy": sum(row["correct"] for row in per_cue.values()) / 4,
                "per_cue": per_cue,
                "replay_packets": 4 if arm == "unconditional_two_step" else 2,
                "prediction_trace_sha256": "matched",
            }
        per_seed[str(seed)] = arms
    scored = {"per_seed": per_seed, "oracle_checks": {"exact": True},
              "resource_checks": {"bounded": True}}
    assert module["decision"](protocol, scored)["heldout_gate_passed"]
    scored["oracle_checks"]["exact"] = False
    assert not module["decision"](protocol, scored)["heldout_gate_passed"]
    scored["oracle_checks"]["exact"] = True
    for seed in per_seed.values():
        seed["selective_two_step"]["replay_packets"] = 4
    assert not module["decision"](protocol, scored)["heldout_gate_passed"]


def test_candidate_oracle_digest_adapter_on_consumed_development_data():
    module, _ = _module_and_protocol()
    candidate = runpy.run_path(str(module["CANDIDATE"]))
    materializer = runpy.run_path(str(module["MATERIALIZER"]))
    parent = json.loads(module["PARENT_PROTOCOL"].read_text())
    seed = parent["identity"]["seeds"][0]
    data = candidate["fixture"](parent, seed)
    seen = {}
    selected = []
    for position, event in enumerate(data["events"], 1):
        original, A_cue, B_cue, A_action, B_action, success = event
        if seen.get(A_cue, 0) < 2:
            seen[A_cue] = seen.get(A_cue, 0) + 1
            selected.append((position, {"A_cue": A_cue, "B_cue": B_cue,
                                        "A_action": A_action, "B_action": B_action,
                                        "success": success, "original_index": original}))
    for arm in parent["arms"]:
        result = candidate["run_arm"](data, arm)
        oracle = materializer["_oracle_arm"](
            selected, data["A_map"], data["B_map"], set(data["wrong_B_cues"]), arm)
        digests = module["_oracle_digests"](
            oracle, selected, set(data["wrong_B_cues"]), data["B_map"],
            arm, data["events"])
        assert result["weight_trace_sha256"] == digests["weights"]
        assert result["prediction_trace_sha256"] == digests["predictions"]
        assert result["forward_trace_sha256"] == digests["forward"]
        assert result["replay_packets"] == oracle["packets"]
