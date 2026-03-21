# Directory Path: src/sara_engine/models/unified_snn.py
# English Title: Unified SNN Model
# Purpose/Content: Single SNN wrapper that coordinates reservoir dynamics, JEPA-style predictive learning, and readout updates without backpropagation.

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .liquid_reservoir import LiquidReservoir
from .spiking_jepa import SpikingJEPA
from .readout_layer import SpikeReadoutLayer


class UnifiedSNNModel:
    """
    Unified SNN model that composes reservoir dynamics, predictive JEPA learning,
    and readout adaptation in a single update schedule.
    """

    def __init__(
        self,
        reservoir: LiquidReservoir,
        jepa: Optional[SpikingJEPA] = None,
        spike_readout: Optional[SpikeReadoutLayer] = None,
    ):
        self.reservoir = reservoir
        self.jepa = jepa
        self.spike_readout = spike_readout
        self.prev_spikes: List[int] = []

    def reset_state(self, reset_readout: bool = False) -> None:
        self.reservoir.reset_dynamic_state(reset_readout=reset_readout)
        if self.jepa is not None:
            self.jepa.reset_state()
        self.prev_spikes = []

    def step(
        self,
        external_currents: Sequence[float],
        *,
        target_readout: Optional[Sequence[float]] = None,
        target_future_spikes: Optional[List[int]] = None,
        readout_target_token: Optional[int] = None,
        learning: bool = True,
        delay_manager: Any = None,
    ) -> Dict[str, Any]:
        """
        Runs a single unified update step.

        Args:
            external_currents: Continuous inputs driving the reservoir.
            target_readout: Optional supervision for FORCE readout.
            target_future_spikes: Optional supervision for JEPA.
            readout_target_token: Optional target class for spike readout layer.
            learning: If True, apply learning rules.
            delay_manager: Optional delay manager for reservoir synapses.
        """
        if len(external_currents) != self.reservoir.n:
            raise ValueError(
                f"Expected {self.reservoir.n} external currents, received {len(external_currents)}."
            )
        current_vector = [float(value) for value in external_currents]
        fired = self.reservoir.step(current_vector, delay_manager=delay_manager)

        jepa_output: Optional[List[int]] = None
        jepa_surprise: Optional[float] = None
        if self.jepa is not None and target_future_spikes is not None:
            jepa_output, jepa_surprise = self.jepa.forward(
                x_spikes=self.prev_spikes,
                y_spikes=target_future_spikes,
                learning=learning,
            )

        readout_output: Optional[Any] = None
        if self.reservoir.force_readout is not None:
            reservoir_state = self.reservoir.get_reservoir_state()
            if learning and target_readout is not None:
                readout_output = self.reservoir.force_readout.update(
                    reservoir_state,
                    target_readout,
                )
            else:
                readout_output = self.reservoir.force_readout.predict(
                    reservoir_state,
                )
        elif self.spike_readout is not None:
            readout_output = self.spike_readout.forward(
                fired,
                target_token=readout_target_token,
                learning=learning,
            )

        self.prev_spikes = fired

        return {
            "spikes": fired,
            "jepa_prediction": jepa_output,
            "jepa_surprise": jepa_surprise,
            "readout": readout_output,
        }

    def run_sequence(
        self,
        input_sequence: Sequence[Sequence[float]],
        *,
        target_readouts: Optional[Sequence[Sequence[float]]] = None,
        target_future_spikes: Optional[Sequence[List[int]]] = None,
        readout_target_tokens: Optional[Sequence[Optional[int]]] = None,
        learning: bool = True,
        delay_manager: Any = None,
        reset_state_before_run: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Runs a sequence through the unified model.
        """
        if reset_state_before_run:
            self.reset_state(reset_readout=False)

        sequence_length = len(input_sequence)
        self._validate_optional_sequence_length("target_readouts", target_readouts, sequence_length)
        self._validate_optional_sequence_length("target_future_spikes", target_future_spikes, sequence_length)
        self._validate_optional_sequence_length("readout_target_tokens", readout_target_tokens, sequence_length)

        outputs: List[Dict[str, Any]] = []
        for step_idx, currents in enumerate(input_sequence):
            outputs.append(
                self.step(
                    currents,
                    target_readout=target_readouts[step_idx] if target_readouts else None,
                    target_future_spikes=target_future_spikes[step_idx] if target_future_spikes else None,
                    readout_target_token=readout_target_tokens[step_idx] if readout_target_tokens else None,
                    learning=learning,
                    delay_manager=delay_manager,
                )
            )
        return outputs

    def _validate_optional_sequence_length(
        self,
        name: str,
        values: Optional[Sequence[Any]],
        expected_length: int,
    ) -> None:
        if values is None:
            return
        if len(values) != expected_length:
            raise ValueError(
                f"{name} length must match input_sequence length "
                f"({expected_length}), received {len(values)}."
            )
