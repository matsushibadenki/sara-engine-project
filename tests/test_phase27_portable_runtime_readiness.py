from __future__ import annotations

from sara_engine.edge.canonical_sparse_ir import migrate_state, replay_digest


def test_canonical_sparse_ir_is_order_independent():
    left = [{"event_id": "a", "timestep": 1, "channel": "x", "spike_id": 1, "modality": "vision"}]
    right = list(reversed(left))
    assert replay_digest(left) == replay_digest(right)


def test_canonical_sparse_ir_rejects_unknown_migration():
    state = {"schema": "sara-canonical-ir-state-v1", "events": []}
    try:
        migrate_state(state, from_version="v0", to_version="v1")
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("unknown migration must be rejected")
