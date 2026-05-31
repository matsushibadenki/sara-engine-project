# Directory Path: src/sara_engine/learning/force.py
# English Title: FORCE Readout for Spiking Reservoirs
# Purpose/Content: Provides an online RLS-based readout that can be trained on low-pass filtered SNN reservoir states without backpropagation.

from typing import List, Sequence


class ForceReadout:
    """Online FORCE/RLS readout for reservoir computing."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        alpha: float = 1.0,
        forgetting_factor: float = 1.0,
        weight_clip: float = 10.0,
    ):
        if input_size <= 0:
            raise ValueError('input_size must be positive')
        if output_size <= 0:
            raise ValueError('output_size must be positive')
        if alpha <= 0.0:
            raise ValueError('alpha must be positive')
        if not 0.0 < forgetting_factor <= 1.0:
            raise ValueError('forgetting_factor must be in (0, 1]')
        if weight_clip <= 0.0:
            raise ValueError('weight_clip must be positive')

        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.forgetting_factor = forgetting_factor
        self.weight_clip = weight_clip

        self.weights: List[List[float]] = [
            [0.0 for _ in range(input_size)] for _ in range(output_size)
        ]
        self.bias: List[float] = [0.0 for _ in range(output_size)]
        scale = 1.0 / alpha
        self.inverse_correlation: List[List[float]] = [
            [scale if i == j else 0.0 for j in range(input_size)]
            for i in range(input_size)
        ]

    def predict(self, state: Sequence[float]) -> List[float]:
        self._validate_state(state)
        outputs = []
        for out_idx in range(self.output_size):
            total = self.bias[out_idx]
            row = self.weights[out_idx]
            for idx, value in enumerate(state):
                total += row[idx] * value
            outputs.append(total)
        return outputs

    def update(self, state: Sequence[float], target: Sequence[float]) -> List[float]:
        self._validate_state(state)
        if len(target) != self.output_size:
            raise ValueError('target length must match output_size')

        prediction = self.predict(state)
        gain_vector = self._compute_gain_vector(state)
        error = [prediction[i] - float(target[i]) for i in range(self.output_size)]

        for out_idx in range(self.output_size):
            correction = error[out_idx]
            row = self.weights[out_idx]
            for state_idx in range(self.input_size):
                updated = row[state_idx] - correction * gain_vector[state_idx]
                row[state_idx] = max(-self.weight_clip, min(self.weight_clip, updated))
            self.bias[out_idx] -= correction * 0.05

        self._update_inverse_correlation(state, gain_vector)
        return prediction

    def reset(self) -> None:
        for out_idx in range(self.output_size):
            for state_idx in range(self.input_size):
                self.weights[out_idx][state_idx] = 0.0
            self.bias[out_idx] = 0.0
        scale = 1.0 / self.alpha
        for i in range(self.input_size):
            row = self.inverse_correlation[i]
            for j in range(self.input_size):
                row[j] = scale if i == j else 0.0

    def _compute_gain_vector(self, state: Sequence[float]) -> List[float]:
        projected = [0.0 for _ in range(self.input_size)]
        for i in range(self.input_size):
            total = 0.0
            row = self.inverse_correlation[i]
            for j, value in enumerate(state):
                total += row[j] * value
            projected[i] = total

        denom = self.forgetting_factor
        for i, value in enumerate(state):
            denom += value * projected[i]
        denom = max(1e-9, denom)

        return [value / denom for value in projected]

    def _update_inverse_correlation(self, state: Sequence[float], gain_vector: Sequence[float]) -> None:
        state_projection = [0.0 for _ in range(self.input_size)]
        for j in range(self.input_size):
            total = 0.0
            for i, state_value in enumerate(state):
                total += state_value * self.inverse_correlation[i][j]
            state_projection[j] = total

        new_matrix = [row[:] for row in self.inverse_correlation]
        for i in range(self.input_size):
            for j in range(self.input_size):
                adjusted = new_matrix[i][j] - gain_vector[i] * state_projection[j]
                new_matrix[i][j] = adjusted / self.forgetting_factor
        self.inverse_correlation = new_matrix

    def _validate_state(self, state: Sequence[float]) -> None:
        if len(state) != self.input_size:
            raise ValueError('state length must match input_size')
