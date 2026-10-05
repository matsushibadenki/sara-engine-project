"""Research checkpoint bounds reject malformed weights before replay."""
import pytest
from sara_engine.evaluation.shared_observer_checkpoint import checked_weights


def test_canonical_signed_weights():
    assert checked_weights({'0': -1, '96': 1.0}) == {0: -1, 96: 1.0}


@pytest.mark.parametrize('values', [{'97': 0}, {'00': 0}, {'0': True}, {'0': float('nan')},
                                   {'0': 1.1}, {'-1': 0}, {'１': 0}, []])
def test_bad_weight_state_fails_closed(values):
    with pytest.raises(ValueError):
        checked_weights(values)
