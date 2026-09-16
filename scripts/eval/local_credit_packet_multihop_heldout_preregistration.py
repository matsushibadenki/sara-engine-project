#!/usr/bin/env python3
"""Validate the held-out contract without creating held-out rows."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_multihop_heldout_v1.json"
))
EXPECTED_SHA256 = "5b542db726cf652c69158fa15cb007c3afa06ff615016a201bde9c809470402f"
DEVELOPMENT = Path(workspace_path("evaluation", "local_credit_packet_multihop_development.json"))
HELDOUT_RESULT = Path(workspace_path("evaluation", "local_credit_packet_multihop_heldout_result_v1.json"))


def validate() -> dict[str, object]:
    raw = PROTOCOL.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Frozen multi-hop held-out protocol changed")
    protocol = json.loads(raw)
    sources = protocol["frozen_sources"]
    identity = protocol["identity"]
    boundary = protocol["execution_boundary"]
    source_digests = {
        name: hashlib.sha256((ROOT / sources[f"{name}_path"]).read_bytes()).hexdigest()
        for name in ("candidate", "packet")
    }
    development_digest = hashlib.sha256(DEVELOPMENT.read_bytes()).hexdigest()
    checks = {
        "schema": protocol["schema"]
        == "sara-local-credit-packet-multihop-heldout-preregistration-v1",
        "development_frozen": development_digest == protocol["development_result_sha256"],
        "candidate_frozen": source_digests["candidate"] == sources["candidate_sha256"],
        "packet_frozen": source_digests["packet"] == sources["packet_sha256"],
        "independent_generator": protocol["independence"]["new_generator_required"]
        and not protocol["independence"]["prior_multihop_generator_import_allowed"]
        and not protocol["independence"]["prior_multihop_episode_rows_reused"],
        "separate_oracle": protocol["independence"]["separate_label_evaluator_required"],
        "fresh_seeds": len(identity["seeds"]) == 5
        and not set(identity["seeds"]) & {977021, 977123, 977227, 977329, 977431},
        "fresh_cues": len(identity["A_cues"]) == len(identity["B_cues"]) == 8
        and not (set(identity["A_cues"]) | set(identity["B_cues"])) & set(range(8))
        and not set(identity["A_cues"]) & set(identity["B_cues"]),
        "balanced_counts": identity["B_calibration_rows_per_seed"] == 128
        and identity["main_training_rows_per_seed"] == 512
        and identity["heldout_rows_per_seed"] == 128,
        "five_arms": protocol["arms"] == [
            "no_A_credit", "global_outcome_broadcast", "direct_anchor_shortcut",
            "local_credit_packet", "oracle_control",
        ],
        "one_shot": boundary["maximum_heldout_executions"] == 1,
        "candidate_immutable": boundary["candidate_changes_after_materialization_allowed"] is False,
        "zero_shot": boundary["heldout_updates"] is False,
        "heldout_unopened": identity["heldout_materialized"] is False
        and identity["heldout_consumed"] is False,
        "production_closed": boundary["production_authorized"] is False,
    }
    if not all(checks.values()):
        raise ValueError("Held-out multi-hop preregistration invalid")
    report = {
        "schema": "sara-local-credit-packet-multihop-heldout-validation-v1",
        "protocol_sha256": digest,
        "development_result_sha256": development_digest,
        "source_sha256": source_digests,
        "checks": checks,
        "passed": True,
        "heldout_result_exists_now": HELDOUT_RESULT.exists(),
        "heldout_consumed": HELDOUT_RESULT.exists(),
        "production_authorized": False,
    }
    output = Path(ensure_parent_directory(workspace_path(
        "evaluation", "local_credit_packet_multihop_heldout_preregistration.json"
    )))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), **report}


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
