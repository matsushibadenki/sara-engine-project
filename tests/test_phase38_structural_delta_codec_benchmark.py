from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def _module():
    path=ROOT/"scripts"/"eval"/"phase38_structural_delta_codec_benchmark.py"; spec=importlib.util.spec_from_file_location("phase38_codec_benchmark",path); assert spec and spec.loader; module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_phase38_codec_exactness_passes_but_malformed_gate_is_retained_negative():
    module=_module(); report=module.build_report(module._rows(module.INPUTS),module._rows(module.KEY))
    assert report["metrics"]["exact_reconstruction_rate"]==1.0
    assert report["metrics"]["digest_match_rate"]==1.0
    assert report["metrics"]["rollback_fidelity"]==1.0
    assert report["metrics"]["provenance_tombstone_preservation"]==1.0
    assert report["checks"]["malformed_abstention_gate"] is False
    assert report["passed"] is False and report["retained_negative_result"] is True and report["promotion_ready"] is False
    assert report["transformation_sharing_executed"] is False


def test_phase38_candidate_traces_do_not_contain_evaluator_labels():
    module=_module(); report=module.build_report(module._rows(module.INPUTS),module._rows(module.KEY)); trace=json.dumps([row["candidate"] for row in report["results"]],sort_keys=True)
    for forbidden in ("case_family","expected_decision","exact_target","withheld_delta"): assert forbidden not in trace
