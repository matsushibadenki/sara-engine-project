from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "scripts" / "eval" / "phase34_transcribed_excerpt_review_request.py"
    spec = importlib.util.spec_from_file_location(
        "phase34_transcribed_excerpt_review_request", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_request_binds_six_pending_reviews_without_mutating_history():
    module = _load_module()
    request = module.build_request(
        module._read_jsonl(module.DEFAULT_RAW),
        module._read_json(module.DEFAULT_PROVENANCE),
    )

    assert request["target_count"] == 6
    assert request["review_complete"] is False
    assert request["promotion_ready"] is False
    assert all(
        target["review_status"] == "pending_human_review"
        and target["required_review"]["alignment_decision"] == "pending"
        for target in request["targets"]
    )
    assert request["mutation_policy"] == {
        "historical_raw_rows_mutated": False,
        "executed_v2_fingerprint_mutated": False,
        "silent_reclassification_allowed": False,
        "replacement_requires_new_preregistration": True,
    }


def test_request_rejects_tampered_excerpt_content():
    module = _load_module()
    rows = module._read_jsonl(module.DEFAULT_RAW)
    rows = copy.deepcopy(rows)
    target_ids = {
        row["record_id"]
        for row in module._read_json(module.DEFAULT_PROVENANCE)["manual_review_targets"]
    }
    target = next(row for row in rows if row.get("record_id") in target_ids)
    target["content"] += " tampered"

    with pytest.raises(ValueError, match="content binding"):
        module.build_request(rows, module._read_json(module.DEFAULT_PROVENANCE))
