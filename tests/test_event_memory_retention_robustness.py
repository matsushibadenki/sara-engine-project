"""Evaluator-isolated retention robustness experiment."""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval/event_memory_retention_robustness.py"
SPEC = importlib.util.spec_from_file_location("event_memory_retention_robustness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _inputs():
    protocol = json.loads(MODULE.PROTOCOL.read_text())
    base_path = ROOT / protocol["base_protocol"]
    base = json.loads(base_path.read_text())
    dependency = MODULE._load_module(ROOT / protocol["candidate_source"])
    rows = dependency.load_rows(ROOT / base["manifest"])
    return protocol, base, dependency, rows


def test_retention_robustness_is_bounded_and_evaluator_isolated():
    assert MODULE.digest(MODULE.PROTOCOL) == MODULE.PROTOCOL_SHA256
    protocol, base, dependency, rows = _inputs()
    report = MODULE.build_report(rows, base, protocol, dependency)

    assert report["passed"] is True
    assert report["ordering_count"] == 12
    assert report["integrity_checks"] == {
        "ordering_count": True,
        "distinct_orderings": True,
        "equal_state": True,
        "equal_recall": True,
        "equal_cost": True,
        "evaluator_relevance_isolated": True,
    }
    assert report["promotion_gate_passed"] is False
    assert report["delayed_ordering_wins"] == 3
    assert report["delayed_utility_margin"] < 0.0
    assert report["production_policy_changed"] is False


def test_retention_robustness_result_is_deterministic():
    protocol, base, dependency, rows = _inputs()
    assert MODULE.build_report(rows, base, protocol, dependency) == MODULE.build_report(
        rows, base, protocol, dependency
    )
