from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from sara_engine.edge.canonical_sparse_ir import (
    IR_VERSION,
    STATE_SCHEMA,
    canonical_json,
    canonicalize_events,
    migrate_state,
    replay_digest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_readiness_module():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "eval"
        / "phase27_portable_runtime_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase27_portable_runtime_readiness",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_sparse_ir_is_order_independent():
    left = [
        {
            "event_id": "b",
            "timestep": 1,
            "channel": "audio",
            "spike_id": 2,
            "modality": "audio",
            "tags": ["source:b"],
        },
        {
            "event_id": "a",
            "timestep": 1,
            "channel": "vision",
            "spike_id": 1,
            "modality": "vision",
            "confidence": 0.12345649,
            "tags": ["source:a", "source:a"],
        },
    ]
    right = list(reversed(left))
    assert replay_digest(left) == replay_digest(right)
    assert canonicalize_events(left)[0].event_id == "a"
    assert '"confidence":0.123456' in canonical_json(left)
    assert '"tags":["source:a"]' in canonical_json(left)


def test_canonical_sparse_ir_rejects_unknown_migration():
    state = {"schema": STATE_SCHEMA, "ir_version": IR_VERSION, "events": []}
    with pytest.raises(ValueError, match="unsupported"):
        migrate_state(state, from_version="v0", to_version="v1")


def test_canonical_sparse_ir_migration_is_idempotent_and_version_bound():
    state = {
        "schema": STATE_SCHEMA,
        "ir_version": IR_VERSION,
        "events": [
            {
                "event_id": "a",
                "timestep": 0,
                "channel": "x",
                "spike_id": 1,
                "modality": "vision",
            }
        ],
    }
    migrated = migrate_state(
        state, from_version=IR_VERSION, to_version=IR_VERSION
    )
    assert (
        migrate_state(
            migrated, from_version=IR_VERSION, to_version=IR_VERSION
        )
        == migrated
    )
    mismatched = dict(state, ir_version="other")
    with pytest.raises(ValueError, match="does not match"):
        migrate_state(
            mismatched, from_version=IR_VERSION, to_version=IR_VERSION
        )


@pytest.mark.parametrize(
    ("event_patch", "message"),
    [
        ({"event_id": ""}, "non-empty string"),
        ({"timestep": -1}, "non-negative integer"),
        ({"spike_id": True}, "non-negative integer"),
        ({"confidence": math.nan}, "finite number"),
        ({"confidence": 1.1}, "between 0.0 and 1.0"),
        ({"tags": "not-a-list"}, "list or tuple"),
        ({"unknown": "field"}, "unknown fields"),
    ],
)
def test_canonical_sparse_ir_rejects_invalid_events(event_patch, message):
    event = {
        "event_id": "a",
        "timestep": 1,
        "channel": "x",
        "spike_id": 1,
        "modality": "vision",
    }
    event.update(event_patch)
    with pytest.raises(ValueError, match=message):
        replay_digest([event])


def test_canonical_sparse_ir_rejects_duplicate_ids_and_event_overflow():
    event = {
        "event_id": "a",
        "timestep": 1,
        "channel": "x",
        "spike_id": 1,
        "modality": "vision",
    }
    with pytest.raises(ValueError, match="duplicate event_id"):
        canonicalize_events([event, event])
    with pytest.raises(ValueError, match="event count exceeds"):
        canonicalize_events([event, dict(event, event_id="b")], max_events=1)


def test_canonical_sparse_ir_frozen_multimodal_digest():
    events = [
        {
            "event_id": "vision-1",
            "timestep": 1,
            "channel": "vision",
            "spike_id": 7,
            "modality": "vision",
            "confidence": 0.875,
            "tags": ["source:camera-a", "object:door"],
        },
        {
            "event_id": "audio-1",
            "timestep": 2,
            "channel": "audio",
            "spike_id": 11,
            "modality": "audio",
            "confidence": 0.625,
            "tags": ["source:microphone-a"],
        },
    ]
    assert replay_digest(events) == (
        "b66fdf601d0c3ab44e648995bbb70ef1675a2d30c61c6d6b294d243f183db18b"
    )


def test_phase27_readiness_surfaces_observed_tokenizer_conformance():
    module = _load_readiness_module()
    tokenizer_report = {
        "passed": True,
        "observed_only": True,
        "production_path_changed": False,
        "rust_path_observed": True,
        "rust_scalar_reference_available": True,
        "rust_scalar_reference_equivalent": True,
        "checks": {
            "token_ids_equivalent": True,
            "decode_round_trip_preserved": True,
            "spike_event_digest_equivalent": True,
        },
    }

    report = module.build_report(tokenizer_report=tokenizer_report)

    assert report["passed"] is True
    assert report["tokenizer_acceleration_conformance_observed"] is True
    assert report["rust_scalar_tokenizer_equivalence_observed"] is True
    assert report["tokenizer_acceleration_production_promoted"] is False


def test_phase27_readiness_blocks_unreviewed_tokenizer_promotion():
    module = _load_readiness_module()
    tokenizer_report = {
        "passed": True,
        "observed_only": True,
        "production_path_changed": True,
        "checks": {
            "token_ids_equivalent": True,
            "decode_round_trip_preserved": True,
            "spike_event_digest_equivalent": True,
        },
    }

    report = module.build_report(tokenizer_report=tokenizer_report)

    assert report["passed"] is False
    assert report["checks"]["tokenizer_acceleration_not_promoted"] is False
