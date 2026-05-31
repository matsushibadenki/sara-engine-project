from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .direct_map import DirectMap, restore_direct_map


@dataclass(frozen=True)
class TurboQuantConfig:
    main_bits: int = 3
    residual_scale: float | None = None
    enable_smoothing: bool = True


class HybridTurboQuantEngine:
    """
    Lightweight hybrid quantization for scalar lists.
    It applies local smoothing, low-bit uniform quantization, and a 1-bit residual sign.
    """

    def __init__(self, config: TurboQuantConfig | None = None):
        self.config = config or TurboQuantConfig()
        self.levels = max(2, 2 ** int(self.config.main_bits))

    def _pseudo_polar_quant(self, vector: Sequence[float], apply_smoothing: bool = True) -> List[float]:
        if not self.config.enable_smoothing or not apply_smoothing:
            return [float(value) for value in vector]

        n = len(vector)
        if n == 0:
            return []

        mixed = [0.0] * n
        for i in range(n):
            mixed[i] = (float(vector[i]) + float(vector[(i + 1) % n])) * 0.7071
        return mixed

    def _uniform_quantize(self, value: float, min_v: float, max_v: float) -> int:
        if max_v == min_v:
            return 0
        normalized = (value - min_v) / (max_v - min_v)
        quantized = round(normalized * (self.levels - 1))
        return max(0, min(self.levels - 1, int(quantized)))

    def _uniform_dequantize(self, q_value: int, min_v: float, max_v: float) -> float:
        if self.levels <= 1:
            return min_v
        normalized = float(q_value) / float(self.levels - 1)
        return min_v + normalized * (max_v - min_v)

    @staticmethod
    def _uniform_dequantize_with_levels(q_value: int, min_v: float, max_v: float, levels: int) -> float:
        if levels <= 1:
            return min_v
        normalized = float(q_value) / float(levels - 1)
        return min_v + normalized * (max_v - min_v)

    def _resolve_residual_scale(self, min_v: float, max_v: float, override: float | None = None) -> float:
        if override is not None:
            return float(override)
        if self.config.residual_scale is not None:
            return float(self.config.residual_scale)
        spread = max_v - min_v
        if spread <= 0.0:
            return 0.0
        return spread / max(8.0, float((self.levels - 1) * 4))

    def quantize_vector(self, vector: Sequence[float], apply_smoothing: bool = True) -> Dict[str, object]:
        smoothed_vector = self._pseudo_polar_quant(vector, apply_smoothing=apply_smoothing)
        if not smoothed_vector:
            return {
                "main_q": [],
                "residual_q": [],
                "min_val": 0.0,
                "max_val": 0.0,
                "res_scale": 0.0,
            }

        min_val = min(smoothed_vector)
        max_val = max(smoothed_vector)
        res_scale = self._resolve_residual_scale(min_val, max_val)

        main_quantized: List[int] = []
        residual_1bit: List[int] = []
        for val in smoothed_vector:
            q_val = self._uniform_quantize(val, min_val, max_val)
            main_quantized.append(q_val)

            approx_val = self._uniform_dequantize(q_val, min_val, max_val)
            error = val - approx_val
            residual_1bit.append(1 if error > 0.0 else 0)

        return {
            "main_q": main_quantized,
            "residual_q": residual_1bit,
            "min_val": float(min_val),
            "max_val": float(max_val),
            "res_scale": float(res_scale),
            "main_bits": int(self.config.main_bits),
        }

    @staticmethod
    def _coerce_int_list(values: object) -> List[int]:
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes, bytearray, dict)):
            return []
        result: List[int] = []
        for value in values:
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _coerce_float(value: object, default: float = 0.0) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float, str, bytes, bytearray)):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        return default

    @staticmethod
    def _coerce_int(value: object, default: int) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float, str, bytes, bytearray)):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
        return default

    def reconstruct_vector(self, payload: Mapping[str, object]) -> List[float]:
        main_q = self._coerce_int_list(payload.get("main_q", []))
        residual_q = self._coerce_int_list(payload.get("residual_q", []))
        min_v = self._coerce_float(payload.get("min_val", 0.0), 0.0)
        max_v = self._coerce_float(payload.get("max_val", 0.0), 0.0)
        res_scale = self._coerce_float(payload.get("res_scale", 0.0), 0.0)
        main_bits = self._coerce_int(payload.get("main_bits", self.config.main_bits), int(self.config.main_bits))
        levels = max(2, 2 ** main_bits)

        reconstructed: List[float] = []
        for mq, rq in zip(main_q, residual_q):
            base_val = self._uniform_dequantize_with_levels(mq, min_v, max_v, levels)
            correction = res_scale if rq == 1 else -res_scale
            reconstructed.append(base_val + correction)
        return reconstructed

    def quantize_weight_row(self, weights: Mapping[int, float]) -> Dict[str, object]:
        ordered = sorted((int(token_id), float(weight)) for token_id, weight in weights.items())
        token_ids = [token_id for token_id, _ in ordered]
        vector = [weight for _, weight in ordered]
        payload = self.quantize_vector(vector, apply_smoothing=False)
        payload["token_ids"] = token_ids
        return payload

    def reconstruct_weight_row(self, payload: Mapping[str, object]) -> Dict[int, float]:
        token_ids = self._coerce_int_list(payload.get("token_ids", []))
        vector = self.reconstruct_vector(payload)
        return {token_id: float(weight) for token_id, weight in zip(token_ids, vector)}

    def quantize_direct_map(self, direct_map: Mapping[Tuple[int, ...], Mapping[int, float]]) -> Dict[str, Dict[str, object]]:
        quantized: Dict[str, Dict[str, object]] = {}
        for key, row in direct_map.items():
            quantized[str(tuple(int(item) for item in key))] = self.quantize_weight_row(row)
        return quantized

    def restore_direct_map(self, payload: Mapping[object, Mapping[str, object]]) -> DirectMap:
        restored: DirectMap = {}
        parsed_keys = restore_direct_map({raw_key: {} for raw_key in payload.keys()})
        for key in parsed_keys.keys():
            raw_row = payload.get(str(key), payload.get(key))  # type: ignore[arg-type]
            if raw_row is None:
                continue
            restored[key] = self.reconstruct_weight_row(raw_row)
        return restored

    def quantize_readout_synapses(
        self,
        readout_synapses: Sequence[Mapping[int, Tuple[float, int]]],
    ) -> List[Dict[str, object]]:
        quantized_rows: List[Dict[str, object]] = []
        for row in readout_synapses:
            ordered = sorted((int(post_id), float(weight), int(branch_id)) for post_id, (weight, branch_id) in row.items())
            token_ids = [post_id for post_id, _, _ in ordered]
            branch_ids = [branch_id for _, _, branch_id in ordered]
            payload = self.quantize_vector([weight for _, weight, _ in ordered], apply_smoothing=False)
            payload["token_ids"] = token_ids
            payload["branch_ids"] = branch_ids
            quantized_rows.append(payload)
        return quantized_rows

    def restore_readout_synapses(
        self,
        payload: Sequence[Mapping[str, object]],
    ) -> List[Dict[int, Tuple[float, int]]]:
        restored: List[Dict[int, Tuple[float, int]]] = []
        for row in payload:
            token_ids = self._coerce_int_list(row.get("token_ids", []))
            branch_ids = self._coerce_int_list(row.get("branch_ids", []))
            weights = self.reconstruct_vector(row)
            restored.append({
                token_id: (float(weight), branch_id)
                for token_id, weight, branch_id in zip(token_ids, weights, branch_ids)
            })
        return restored

    def quantize_weight_rows(
        self,
        rows: Sequence[Mapping[int, float]],
    ) -> List[Dict[str, object]]:
        quantized_rows: List[Dict[str, object]] = []
        for row in rows:
            quantized_rows.append(self.quantize_weight_row(row))
        return quantized_rows

    def restore_weight_rows(
        self,
        payload: Sequence[Mapping[str, object]],
    ) -> List[Dict[int, float]]:
        restored: List[Dict[int, float]] = []
        for row in payload:
            restored.append(self.reconstruct_weight_row(row))
        return restored

    def quantize_delay_synapses(
        self,
        synapses: Mapping[int, Mapping[int, Mapping[int, float]]],
    ) -> Dict[str, Dict[str, Dict[str, object]]]:
        quantized: Dict[str, Dict[str, Dict[str, object]]] = {}
        for delay, pre_dict in synapses.items():
            delay_payload: Dict[str, Dict[str, object]] = {}
            for pre_id, post_dict in pre_dict.items():
                delay_payload[str(int(pre_id))] = self.quantize_weight_row(post_dict)
            quantized[str(int(delay))] = delay_payload
        return quantized

    def restore_delay_synapses(
        self,
        payload: Mapping[object, Mapping[object, Mapping[str, object]]],
    ) -> Dict[int, Dict[int, Dict[int, float]]]:
        restored: Dict[int, Dict[int, Dict[int, float]]] = {}
        for raw_delay, raw_pre_dict in payload.items():
            delay = self._coerce_int(raw_delay, 0)
            restored[delay] = {}
            for raw_pre_id, row_payload in raw_pre_dict.items():
                pre_id = self._coerce_int(raw_pre_id, 0)
                restored[delay][pre_id] = self.reconstruct_weight_row(row_payload)
        return restored


def create_turboquant_engine(
    main_bits: int = 3,
    residual_scale: float | None = None,
    metadata: Mapping[str, object] | None = None,
    *,
    enable_smoothing: bool = True,
) -> HybridTurboQuantEngine:
    resolved_main_bits = (
        HybridTurboQuantEngine._coerce_int(metadata.get("main_bits", main_bits), int(main_bits))
        if metadata
        else int(main_bits)
    )
    if metadata and "residual_scale" in metadata:
        raw_residual_scale = metadata.get("residual_scale")
        resolved_residual_scale = None if raw_residual_scale is None else HybridTurboQuantEngine._coerce_float(raw_residual_scale)
    else:
        resolved_residual_scale = residual_scale
    return HybridTurboQuantEngine(
        TurboQuantConfig(
            main_bits=resolved_main_bits,
            residual_scale=resolved_residual_scale,
            enable_smoothing=enable_smoothing,
        )
    )


def turboquant_metadata(engine: HybridTurboQuantEngine) -> Dict[str, object]:
    return {
        "format": "turboquant",
        "main_bits": int(engine.config.main_bits),
        "residual_scale": engine.config.residual_scale,
    }
