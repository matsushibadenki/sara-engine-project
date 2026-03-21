# Directory Path: src/sara_engine/learning/force_io.py
# English Title: FORCE Artifact IO
# Purpose/Content: Saves and restores FORCE reservoir artifacts so trained readouts can be reused for inference-only workflows.

import json
import os
import shutil
from typing import Any, Dict, Tuple, TYPE_CHECKING

from ..encoders.time_series import TimeSeriesCurrentEncoder
from ..utils.project_paths import ensure_output_directory, ensure_parent_directory

if TYPE_CHECKING:
    from ..models.liquid_reservoir import LiquidReservoir


def export_force_artifact(
    artifact_path: str,
    report_path: str,
    reservoir: "LiquidReservoir",
    encoder: TimeSeriesCurrentEncoder,
    metadata: Dict[str, Any],
) -> Tuple[str, str]:
    if reservoir.force_readout is None:
        raise RuntimeError("FORCE readout must be enabled before exporting artifacts.")

    ensured_artifact_path = ensure_parent_directory(artifact_path)
    ensured_report_path = ensure_parent_directory(report_path)
    artifact_parent = os.path.dirname(ensured_artifact_path)
    if artifact_parent:
        ensure_output_directory(artifact_parent)

    artifact_payload = {
        "metadata": metadata,
        "encoder": {
            "amplitude": encoder.amplitude,
            "delta_scale": encoder.delta_scale,
            "quadratic_scale": encoder.quadratic_scale,
            "magnitude_scale": encoder.magnitude_scale,
            "band_growth": encoder.band_growth,
        },
        "reservoir": {
            "n_neurons": reservoir.n,
            "dt": reservoir.dt,
            "max_weight": reservoir.max_weight,
            "max_delay_limit": reservoir.max_delay_limit,
            "readout_decay": reservoir.readout_decay,
            "synapses": [
                {str(post_id): weight for post_id, weight in row.items()}
                for row in reservoir.synapses
            ],
            "is_inhibitory": list(reservoir.is_inhibitory),
        },
        "force_readout": {
            "weights": reservoir.force_readout.weights,
            "bias": reservoir.force_readout.bias,
            "inverse_correlation": reservoir.force_readout.inverse_correlation,
            "alpha": reservoir.force_readout.alpha,
            "forgetting_factor": reservoir.force_readout.forgetting_factor,
            "weight_clip": reservoir.force_readout.weight_clip,
        },
    }

    report_payload = dict(metadata)
    report_payload["artifact_path"] = ensured_artifact_path

    with open(ensured_artifact_path, "w", encoding="utf-8") as handle:
        json.dump(artifact_payload, handle, indent=2, ensure_ascii=False)
    with open(ensured_report_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, ensure_ascii=False)

    latest_artifact_path = os.path.join(artifact_parent, "latest_force_run.json")
    shutil.copyfile(ensured_artifact_path, latest_artifact_path)

    best_artifact_path = os.path.join(artifact_parent, "best_force_run.json")
    current_score = _extract_force_score(metadata)
    should_update_best = True
    if os.path.exists(best_artifact_path):
        try:
            with open(best_artifact_path, "r", encoding="utf-8") as handle:
                best_payload = json.load(handle)
            best_score = _extract_force_score(best_payload.get("metadata", {}))
            should_update_best = current_score <= best_score
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
            should_update_best = True
    if should_update_best:
        shutil.copyfile(ensured_artifact_path, best_artifact_path)

    return ensured_artifact_path, ensured_report_path


def load_force_artifact(artifact_path: str) -> Tuple["LiquidReservoir", TimeSeriesCurrentEncoder, Dict[str, Any]]:
    with open(artifact_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    encoder_payload = payload.get("encoder", {})
    reservoir_payload = payload.get("reservoir", {})
    readout_payload = payload.get("force_readout", {})

    synapses_payload = reservoir_payload.get("synapses")
    if synapses_payload is None:
        raise ValueError(
            "Artifact is missing reservoir synapses. Re-export the model with the updated FORCE artifact format."
        )
    if not isinstance(synapses_payload, list):
        raise ValueError("Artifact reservoir synapses must be stored as a list.")

    n_neurons = int(reservoir_payload["n_neurons"])
    if len(synapses_payload) != n_neurons:
        raise ValueError(
            f"Artifact synapse row count ({len(synapses_payload)}) does not match n_neurons ({n_neurons})."
        )

    weights_payload = readout_payload.get("weights")
    bias_payload = readout_payload.get("bias")
    inverse_payload = readout_payload.get("inverse_correlation")
    if not isinstance(weights_payload, list) or not isinstance(bias_payload, list) or not isinstance(inverse_payload, list):
        raise ValueError("Artifact FORCE readout is incomplete. Expected weights, bias, and inverse_correlation lists.")
    if len(weights_payload) != len(bias_payload):
        raise ValueError("Artifact FORCE readout weights must align with bias dimension.")

    from ..models.liquid_reservoir import LiquidReservoir
    reservoir = LiquidReservoir(
        n_neurons=n_neurons,
        dt=float(reservoir_payload.get("dt", 1.0)),
        max_weight=float(reservoir_payload.get("max_weight", 2.0)),
        max_delay_limit=int(reservoir_payload.get("max_delay_limit", 50)),
        readout_decay=float(reservoir_payload.get("readout_decay", 0.9)),
        enable_force_readout=True,
        force_output_dim=max(1, len(bias_payload)),
        force_alpha=float(readout_payload.get("alpha", 1.0)),
        force_forgetting_factor=float(readout_payload.get("forgetting_factor", 1.0)),
    )

    reservoir.synapses = [
        {int(post_id): float(weight) for post_id, weight in row.items()}
        for row in synapses_payload
    ]
    inhibitory_payload = reservoir_payload.get("is_inhibitory")
    if inhibitory_payload is not None and len(inhibitory_payload) == reservoir.n:
        reservoir.is_inhibitory = [bool(value) for value in inhibitory_payload]

    if reservoir.force_readout is None:
        raise RuntimeError("Failed to create FORCE readout during artifact restore.")

    expected_input_size = reservoir.force_readout.input_size
    for row in weights_payload:
        if not isinstance(row, list) or len(row) != expected_input_size:
            raise ValueError(
                f"Artifact FORCE weights must contain rows of length {expected_input_size}."
            )
    if len(inverse_payload) != expected_input_size or any(
        not isinstance(row, list) or len(row) != expected_input_size for row in inverse_payload
    ):
        raise ValueError(
            f"Artifact inverse_correlation must be a square matrix of size {expected_input_size}."
        )

    reservoir.force_readout.weights = [
        [float(value) for value in row]
        for row in weights_payload
    ]
    reservoir.force_readout.bias = [float(value) for value in bias_payload]
    reservoir.force_readout.inverse_correlation = [
        [float(value) for value in row]
        for row in inverse_payload
    ]

    encoder = TimeSeriesCurrentEncoder(
        amplitude=float(encoder_payload.get("amplitude", 10.0)),
        delta_scale=float(encoder_payload.get("delta_scale", 1.4)),
        quadratic_scale=float(encoder_payload.get("quadratic_scale", 0.7)),
        magnitude_scale=float(encoder_payload.get("magnitude_scale", 0.5)),
        band_growth=float(encoder_payload.get("band_growth", 0.15)),
    )

    metadata = payload.get("metadata", {})
    return reservoir, encoder, metadata


def _extract_force_score(metadata: Dict[str, Any]) -> float:
    for key in ("final_test_mae", "baseline_test_mae"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float("inf")
