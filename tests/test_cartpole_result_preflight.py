"""Regression tests for the V1 NumPy scalar serialization failure."""
import json

import numpy as np
import pytest

from sara_engine.evaluation.cartpole_result_preflight import (
    prepare_environment_audit, preflight_result_envelope,
)
from sara_engine.evaluation.cartpole_sparse_reward import inspect_environment


def test_live_unscored_audit_is_normalized_before_result_serialization():
    audit = inspect_environment()
    assert isinstance(audit["action_count"], np.integer)
    with pytest.raises(TypeError):
        json.dumps(audit)
    normalized = prepare_environment_audit(audit)
    assert type(normalized["action_count"]) is int
    assert json.loads(json.dumps(normalized))["action_count"] == 2
    preflight_result_envelope(audit)


def test_preflight_rejects_wrong_audit_without_normalizing_arbitrary_values():
    audit = inspect_environment()
    audit["action_count"] = 2.0
    with pytest.raises(ValueError, match="integral"):
        preflight_result_envelope(audit)
    audit["action_count"] = 2
    audit["evaluator_invoked"] = True
    with pytest.raises(ValueError, match="does not match"):
        preflight_result_envelope(audit)
