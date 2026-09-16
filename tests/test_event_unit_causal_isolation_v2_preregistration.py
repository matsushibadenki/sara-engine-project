import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval" / "event_unit_causal_isolation_v2_preregistration.py"
SPEC = importlib.util.spec_from_file_location("event_unit_causal_isolation_v2_preregistration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_event_unit_v2_protocol_is_valid():
    result = MODULE.validate()
    assert result["passed"] is True
    assert result["candidate_implemented"] is False
    assert result["held_out_consumed"] is False
    assert all(result["checks"].values())
