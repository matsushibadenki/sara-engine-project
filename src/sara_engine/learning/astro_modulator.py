# Directory Path: src/sara_engine/learning/astro_modulator.py
# English Title: Astrocyte-Inspired Replay Modulator
# Purpose/Content: Lightweight astrocyte-style slow-timescale modulation for replay/consolidation loops without backpropagation or dense matrix operations.

from dataclasses import dataclass


@dataclass
class AstroState:
    stress_level: float = 0.0
    support_level: float = 1.0
    stability_level: float = 1.0


class AstroReplayModulator:
    """Slow-timescale modulation layer for replay/consolidation stability.

    This helper intentionally uses scalar updates only:
    - no backpropagation
    - no dense matrix operations
    - CPU-friendly deterministic state updates
    """

    def __init__(
        self,
        stress_decay: float = 0.90,
        support_recovery: float = 0.05,
        stress_gain: float = 0.60,
    ) -> None:
        self.stress_decay = float(stress_decay)
        self.support_recovery = float(support_recovery)
        self.stress_gain = float(stress_gain)
        self.state = AstroState()

    def update(self, *, interference_ratio: float, replay_recovery_signal: float) -> None:
        clamped_interference = max(0.0, min(1.0, float(interference_ratio)))
        clamped_recovery = max(0.0, min(1.0, float(replay_recovery_signal)))

        stress = self.state.stress_level * self.stress_decay
        stress += clamped_interference * self.stress_gain
        stress -= clamped_recovery * 0.35
        self.state.stress_level = max(0.0, min(1.0, stress))

        support = self.state.support_level + self.support_recovery * clamped_recovery
        support -= 0.20 * self.state.stress_level
        self.state.support_level = max(0.10, min(1.40, support))

        stability = 1.0 - 0.65 * self.state.stress_level + 0.25 * (self.state.support_level - 1.0)
        self.state.stability_level = max(0.0, min(1.2, stability))

    def modulate_replay_weight(self, base_weight: float) -> float:
        return float(base_weight) * self.state.support_level * max(0.40, self.state.stability_level)

    def snapshot(self) -> dict:
        return {
            "stress_level": float(self.state.stress_level),
            "support_level": float(self.state.support_level),
            "stability_level": float(self.state.stability_level),
        }
