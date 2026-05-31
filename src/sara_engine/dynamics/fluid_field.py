# Directory Path: src/sara_engine/dynamics/fluid_field.py
# English Title: Fluid Field Dynamics
# Purpose/Content: Lightweight scalar-only fluid-inspired field dynamics for supplementary context propagation, bounded forgetting, and spike extraction.

from __future__ import annotations

from typing import Dict, List, Sequence


class FluidFieldDynamics:
    """
    A bounded fluid-inspired auxiliary field that uses only scalar/list operations.
    It is designed as a supplementary dynamics layer rather than a standalone model.
    """

    def __init__(
        self,
        width: int = 16,
        height: int = 8,
        clip_value: int = 10,
        spike_threshold: int = 4,
    ) -> None:
        self.width = max(4, int(width))
        self.height = max(4, int(height))
        self.clip_value = max(1, int(clip_value))
        self.spike_threshold = max(1, int(spike_threshold))

    def create_field(self) -> List[List[int]]:
        return [[0 for _ in range(self.width)] for _ in range(self.height)]

    def wave_step(self, field: List[List[int]], prev_field: List[List[int]]) -> List[List[int]]:
        new_field = self.create_field()
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                laplacian = (
                    field[y][x + 1]
                    + field[y][x - 1]
                    + field[y + 1][x]
                    + field[y - 1][x]
                    - (4 * field[y][x])
                )
                viscosity = field[y][x] // 16
                updated = (2 * field[y][x]) - prev_field[y][x] + (laplacian // 2) - viscosity
                if updated > self.clip_value:
                    updated = self.clip_value
                if updated < -self.clip_value:
                    updated = -self.clip_value
                new_field[y][x] = updated
        return new_field

    def vortex_step(self, field: List[List[int]]) -> List[List[int]]:
        new_field = self.create_field()
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                curl = (
                    field[y][x + 1]
                    - field[y][x - 1]
                    + field[y + 1][x]
                    - field[y - 1][x]
                )
                if curl > 2:
                    updated = field[y][x] + 2
                elif curl < -2:
                    updated = field[y][x] - 2
                else:
                    updated = field[y][x]
                if updated > self.clip_value:
                    updated = self.clip_value
                if updated < -self.clip_value:
                    updated = -self.clip_value
                new_field[y][x] = updated
        return new_field

    def snn_step(self, field: List[List[int]]) -> tuple[List[List[int]], List[List[int]]]:
        spikes = self.create_field()
        for y in range(self.height):
            for x in range(self.width):
                if field[y][x] > self.spike_threshold:
                    spikes[y][x] = 1
                    field[y][x] -= self.spike_threshold
                else:
                    spikes[y][x] = 0
        return spikes, field

    def inject_token_ids(self, field: List[List[int]], token_ids: Sequence[int]) -> None:
        for column, token_id in enumerate(token_ids[: self.width]):
            normalized = abs(int(token_id)) % 31
            for row in range(self.height):
                value = ((normalized + 1) * (row + 3) * (column + 1)) % 11
                field[row][column] += value - 5

    def run(self, token_ids: Sequence[int], steps: int = 6) -> Dict[str, object]:
        field = self.create_field()
        prev_field = self.create_field()
        self.inject_token_ids(field, token_ids)

        spike_accum = [0 for _ in range(self.width)]
        peak_amplitude = 0

        for _ in range(max(1, int(steps))):
            new_field = self.wave_step(field, prev_field)
            prev_field = field
            field = self.vortex_step(new_field)
            spikes, field = self.snn_step(field)

            for x in range(self.width):
                column_spikes = 0
                for y in range(self.height):
                    value = abs(field[y][x])
                    if value > peak_amplitude:
                        peak_amplitude = value
                    column_spikes += spikes[y][x]
                spike_accum[x] += column_spikes

        active_columns = sum(1 for value in spike_accum if value > 0)
        total_spikes = sum(spike_accum)
        bounded = peak_amplitude <= self.clip_value
        support_score = min(
            1.0,
            (active_columns / max(self.width // 2, 1)) * 0.5
            + (total_spikes / max(self.width * 2, 1)) * 0.5,
        )

        return {
            "spike_accum": spike_accum,
            "active_columns": active_columns,
            "total_spikes": total_spikes,
            "peak_amplitude": peak_amplitude,
            "bounded": bounded,
            "support_score": float(support_score),
            "width": self.width,
            "height": self.height,
        }
