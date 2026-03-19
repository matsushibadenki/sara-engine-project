# Directory Path: src/sara_engine/encoders/time_series.py
# English Title: Time-Series Current Encoder
# Purpose/Content: Encodes scalar time-series values into reusable multi-channel currents for spiking reservoirs and FORCE readouts.

from typing import List


class TimeSeriesCurrentEncoder:
    """
    Encode a scalar value and its local temporal change into reservoir input currents.
    """

    def __init__(
        self,
        amplitude: float = 10.0,
        delta_scale: float = 1.4,
        quadratic_scale: float = 0.7,
        magnitude_scale: float = 0.5,
        band_growth: float = 0.15,
    ):
        self.amplitude = amplitude
        self.delta_scale = delta_scale
        self.quadratic_scale = quadratic_scale
        self.magnitude_scale = magnitude_scale
        self.band_growth = band_growth

    def encode(
        self,
        value: float,
        previous_value: float,
        n_channels: int,
    ) -> List[float]:
        if n_channels <= 0:
            return []

        currents = [0.0] * n_channels
        delta = value - previous_value
        magnitude = abs(value)

        features = [
            max(0.0, value) * self.amplitude,
            max(0.0, -value) * self.amplitude,
            max(0.0, delta) * self.amplitude * self.delta_scale,
            max(0.0, -delta) * self.amplitude * self.delta_scale,
            value * value * self.amplitude * self.quadratic_scale,
            magnitude * self.amplitude * self.magnitude_scale,
        ]

        feature_count = len(features)
        for channel_idx in range(n_channels):
            feature_idx = channel_idx % feature_count
            band_scale = 1.0 + self.band_growth * (channel_idx // feature_count)
            currents[channel_idx] = features[feature_idx] * band_scale

        return currents
