from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


def _clamp_nonnegative(value: float) -> float:
    return max(0.0, float(value))


@dataclass(frozen=True)
class ChangePoint:
    stream_id: str
    modality: str
    time_ms: int
    value: float
    baseline: float
    delta: float
    threshold: float
    refractory_ms: int
    schema: str = "sara-change-point-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "stream_id": self.stream_id,
            "modality": self.modality,
            "time_ms": int(self.time_ms),
            "value": float(self.value),
            "baseline": float(self.baseline),
            "delta": float(self.delta),
            "threshold": float(self.threshold),
            "refractory_ms": int(self.refractory_ms),
        }


class ScalarChangeDetector:
    """Detects bounded scalar change bursts without ANN assistance."""

    def __init__(
        self,
        *,
        threshold: float = 0.15,
        refractory_ms: int = 80,
        baseline_smoothing: float = 0.8,
    ) -> None:
        self.threshold = _clamp_nonnegative(threshold)
        self.refractory_ms = max(0, int(refractory_ms))
        self.baseline_smoothing = min(0.999, max(0.0, float(baseline_smoothing)))

    def detect(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        stream_id: str,
        modality: str,
    ) -> List[ChangePoint]:
        baseline: float | None = None
        last_change_time = -(10**12)
        changes: List[ChangePoint] = []
        for sample in samples:
            time_ms = int(sample.get("time_ms", 0) or 0)
            value = float(sample.get("value", 0.0) or 0.0)
            if baseline is None:
                baseline = value
                continue
            delta = abs(value - baseline)
            if delta >= self.threshold and (time_ms - last_change_time) >= self.refractory_ms:
                changes.append(
                    ChangePoint(
                        stream_id=str(stream_id),
                        modality=str(modality),
                        time_ms=time_ms,
                        value=value,
                        baseline=baseline,
                        delta=delta,
                        threshold=self.threshold,
                        refractory_ms=self.refractory_ms,
                    )
                )
                last_change_time = time_ms
            baseline = (self.baseline_smoothing * baseline) + ((1.0 - self.baseline_smoothing) * value)
        return changes

