import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.dynamics.fluid_field import FluidFieldDynamics


def test_fluid_field_generates_bounded_activity_summary():
    dynamics = FluidFieldDynamics(width=16, height=8)

    summary = dynamics.run([11, 22, 33, 44, 55], steps=6)

    assert summary["bounded"] is True
    assert summary["active_columns"] >= 1
    assert summary["total_spikes"] > 0
    assert 0.0 < summary["support_score"] <= 1.0
    assert summary["peak_amplitude"] <= 10
