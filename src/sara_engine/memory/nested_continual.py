# Directory Path: src/sara_engine/memory/nested_continual.py
# English Title: Nested Continual Memory Controller
# Purpose/Content: Provides a lightweight multi-rate controller for continuum memory updates without backpropagation or dense matrix dependencies.

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True)
class MemoryUpdateBand:
    name: str
    update_interval: int
    stability_target: float
    energy_cost: float
    transfer_threshold: float


DEFAULT_MEMORY_BANDS: Tuple[MemoryUpdateBand, ...] = (
    MemoryUpdateBand("session", update_interval=1, stability_target=0.20, energy_cost=0.03, transfer_threshold=0.35),
    MemoryUpdateBand("direct", update_interval=2, stability_target=0.45, energy_cost=0.05, transfer_threshold=0.55),
    MemoryUpdateBand("hippocampus", update_interval=4, stability_target=0.65, energy_cost=0.08, transfer_threshold=0.70),
    MemoryUpdateBand("ltm", update_interval=8, stability_target=0.82, energy_cost=0.12, transfer_threshold=0.86),
    MemoryUpdateBand("structural", update_interval=16, stability_target=0.93, energy_cost=0.18, transfer_threshold=0.94),
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class NestedContinualMemoryController:
    """Schedules multi-rate memory updates across a continuum of memory bands."""

    def __init__(
        self,
        bands: Iterable[MemoryUpdateBand] = DEFAULT_MEMORY_BANDS,
        *,
        energy_budget: float = 0.34,
        interference_guard_threshold: float = 0.58,
    ) -> None:
        self.bands = tuple(sorted(bands, key=lambda band: band.update_interval))
        if not self.bands:
            raise ValueError("at least one memory band is required")
        self.energy_budget = float(max(energy_budget, 0.01))
        self.interference_guard_threshold = _clamp01(interference_guard_threshold)
        self.step_index = 0
        self.band_stability: Dict[str, float] = {
            band.name: max(0.05, min(0.98, band.stability_target * 0.8))
            for band in self.bands
        }
        self.update_counts: Dict[str, int] = {band.name: 0 for band in self.bands}
        self.transfer_counts: Dict[str, int] = {band.name: 0 for band in self.bands}
        self.energy_spent = 0.0
        self.guard_events = 0
        self.trace: List[Dict[str, Any]] = []

    def observe(
        self,
        *,
        signal_strength: float,
        interference: float = 0.0,
        novelty: float = 0.0,
        urgency: float = 0.0,
    ) -> Dict[str, Any]:
        signal = _clamp01(signal_strength)
        interference_level = _clamp01(interference)
        novelty_level = _clamp01(novelty)
        urgency_level = _clamp01(urgency)
        self.step_index += 1

        remaining_energy = self.energy_budget
        selected_updates: List[Dict[str, Any]] = []
        selected_transfers: List[Dict[str, Any]] = []
        guard_active = interference_level >= self.interference_guard_threshold
        if guard_active:
            self.guard_events += 1

        for band_index, band in enumerate(self.bands):
            interval_due = self.step_index % max(band.update_interval, 1) == 0
            fast_pressure = urgency_level >= 0.72 and band_index <= 1
            novelty_pressure = novelty_level >= 0.66 and band.name in {"session", "direct", "hippocampus"}
            if not (interval_due or fast_pressure or novelty_pressure):
                self._relax_band(band)
                continue

            if guard_active and band.name in {"ltm", "structural"}:
                self._protect_band(band, interference_level)
                continue

            if remaining_energy < band.energy_cost:
                self._relax_band(band)
                continue

            stability = self.band_stability[band.name]
            learning_gain = signal * (1.0 - 0.45 * interference_level) + novelty_level * 0.12
            stability_delta = max(0.0, learning_gain) * (1.0 - stability) * 0.35
            new_stability = _clamp01(stability + stability_delta)
            self.band_stability[band.name] = new_stability
            self.update_counts[band.name] += 1
            self.energy_spent += band.energy_cost
            remaining_energy -= band.energy_cost
            selected_updates.append(
                {
                    "band": band.name,
                    "stability": new_stability,
                    "energy_cost": band.energy_cost,
                    "reason": self._update_reason(interval_due, fast_pressure, novelty_pressure),
                }
            )

            if new_stability >= band.transfer_threshold:
                self.transfer_counts[band.name] += 1
                selected_transfers.append(
                    {
                        "band": band.name,
                        "target": self._next_band_name(band.name),
                        "stability": new_stability,
                    }
                )

        event = {
            "step": self.step_index,
            "signal_strength": signal,
            "interference": interference_level,
            "novelty": novelty_level,
            "urgency": urgency_level,
            "guard_active": guard_active,
            "updates": selected_updates,
            "transfers": selected_transfers,
            "energy_spent_total": float(self.energy_spent),
        }
        self.trace.append(event)
        return event

    def readiness_metrics(self) -> Dict[str, float]:
        update_counts = [count for count in self.update_counts.values()]
        active_band_count = sum(1 for count in update_counts if count > 0)
        total_updates = sum(update_counts)
        slower_band_updates = sum(
            self.update_counts.get(name, 0)
            for name in ("hippocampus", "ltm", "structural")
        )
        transfer_total = sum(self.transfer_counts.values())
        max_possible_energy = max(self.step_index, 1) * self.energy_budget
        protected_slow_updates = sum(
            1
            for event in self.trace
            if event.get("guard_active")
            for update in event.get("updates", [])
            if isinstance(update, dict) and update.get("band") in {"ltm", "structural"}
        )
        return {
            "multi_rate_update_integrity": 1.0 if active_band_count >= min(4, len(self.bands)) and slower_band_updates > 0 else 0.0,
            "continuum_memory_transfer_stability": 1.0 if transfer_total >= 2 and self.band_stability.get("hippocampus", 0.0) >= 0.70 else 0.0,
            "scheduler_energy_budget_integrity": 1.0 if self.energy_spent <= max_possible_energy else 0.0,
            "catastrophic_interference_guard": 1.0 if self.guard_events > 0 and protected_slow_updates == 0 else 0.0,
            "active_band_ratio": float(active_band_count) / max(len(self.bands), 1),
            "slow_update_ratio": float(slower_band_updates) / max(total_updates, 1),
            "energy_budget_utilization": float(self.energy_spent) / max(max_possible_energy, 1e-9),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "energy_budget": self.energy_budget,
            "energy_spent": float(self.energy_spent),
            "guard_events": self.guard_events,
            "band_stability": dict(self.band_stability),
            "update_counts": dict(self.update_counts),
            "transfer_counts": dict(self.transfer_counts),
            "readiness_metrics": self.readiness_metrics(),
            "trace_tail": self.trace[-8:],
        }

    def _relax_band(self, band: MemoryUpdateBand) -> None:
        stability = self.band_stability[band.name]
        self.band_stability[band.name] = _clamp01(stability * 0.995 + band.stability_target * 0.005)

    def _protect_band(self, band: MemoryUpdateBand, interference: float) -> None:
        stability = self.band_stability[band.name]
        self.band_stability[band.name] = _clamp01(stability - interference * 0.01)

    def _next_band_name(self, current_name: str) -> str:
        names = [band.name for band in self.bands]
        try:
            index = names.index(current_name)
        except ValueError:
            return current_name
        return names[min(index + 1, len(names) - 1)]

    @staticmethod
    def _update_reason(interval_due: bool, fast_pressure: bool, novelty_pressure: bool) -> str:
        reasons = []
        if interval_due:
            reasons.append("interval")
        if fast_pressure:
            reasons.append("urgency")
        if novelty_pressure:
            reasons.append("novelty")
        return "+".join(reasons) if reasons else "scheduled"


def build_nested_memory_report(events: Iterable[Mapping[str, float]]) -> Dict[str, Any]:
    controller = NestedContinualMemoryController()
    event_reports = [
        controller.observe(
            signal_strength=float(event.get("signal_strength", 0.0)),
            interference=float(event.get("interference", 0.0)),
            novelty=float(event.get("novelty", 0.0)),
            urgency=float(event.get("urgency", 0.0)),
        )
        for event in events
    ]
    metrics = controller.readiness_metrics()
    threshold_results = {
        "multi_rate_update_integrity": metrics["multi_rate_update_integrity"] >= 1.0,
        "continuum_memory_transfer_stability": metrics["continuum_memory_transfer_stability"] >= 1.0,
        "scheduler_energy_budget_integrity": metrics["scheduler_energy_budget_integrity"] >= 1.0,
        "catastrophic_interference_guard": metrics["catastrophic_interference_guard"] >= 1.0,
    }
    return {
        "controller_name": "NestedContinualMemoryController",
        "passed": all(threshold_results.values()),
        "metrics": metrics,
        "threshold_results": threshold_results,
        "snapshot": controller.snapshot(),
        "events": event_reports,
    }
