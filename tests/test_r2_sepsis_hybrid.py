from sara_engine.evaluation.r2_sepsis_hybrid import SepsisRouteEncoder, load_protocol
from sara_engine.evaluation.sepsis_data import load_traces

def test_sepsis_protocol_is_bound_and_absolute_time_route_is_absent():
    protocol = load_protocol()
    assert "hour_bucket_x_weekday" not in protocol["feature_routes"]
    assert protocol["inherited_without_tuning"]["base_probability_cap"] == .75

def test_sepsis_encoder_uses_seven_bounded_routes():
    encoder = SepsisRouteEncoder(spiking=False)
    assert len(encoder.encode(load_traces()[0], 0)) == 7
