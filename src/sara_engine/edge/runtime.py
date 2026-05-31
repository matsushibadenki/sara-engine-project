# ディレクトリパス: src/sara_engine/edge/runtime.py
# ファイルの日本語タイトル: Sara-Edge 軽量ランタイム
# ファイルの目的や内容: エッジデバイス向けに最適化された推論専用エンジン。学習用のクラス階層やオーバーヘッドをすべて排除し、最小限のメモリでテキスト生成を実行する。
import hashlib
import json
import random
import operator
from typing import Any, List, Dict, Optional

from ..utils.stochastic_computing import StochasticAccumulator

SUPPORTED_EDGE_FORMAT_VERSION = 2
SUPPORTED_EDGE_CAPABILITIES = {
    "active_row_readout_storage",
    "compact_int_weights",
    "delta_associative_memory_state",
    "delta_event_compression",
    "dense_row_index_space",
    "dense_readout_rows",
    "float_quantized_weights",
    "multilevel_weights",
    "neuromorphic_backend_profile",
    "neuromorphic_spike_event_ir",
    "neuromorphic_state_trace_ir",
    "sparse_routing_table",
}


def _decode_compact_quantized_row(row: Dict[str, Any], quantization_bits: int) -> Dict[int, float]:
    keys = [int(key) for key in row.get("keys", [])]
    qweights = [int(value) for value in row.get("qweights", [])]
    if not keys or not qweights:
        return {}

    levels = max(2, 2 ** int(quantization_bits or 0))
    min_val = float(row.get("min", 0.0))
    max_val = float(row.get("max", min_val))
    if max_val <= min_val:
        return {key: min_val for key in keys[: len(qweights)]}

    decoded: Dict[int, float] = {}
    max_q = levels - 1
    for key, raw_q in zip(keys, qweights):
        q_value = max(0, min(max_q, int(raw_q)))
        decoded[key] = min_val + (q_value / max_q) * (max_val - min_val)
    return decoded


def _decode_delta_ids(deltas: List[int]) -> List[int]:
    decoded: List[int] = []
    current = 0
    for index, delta in enumerate(deltas):
        if index == 0:
            current = int(delta)
        else:
            current += int(delta)
        decoded.append(current)
    return decoded


def _normalize_sparse_routing_table(table: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(table)
    if "active_rows" not in normalized and "active_row_deltas" in normalized:
        normalized["active_rows"] = _decode_delta_ids(
            [int(value) for value in normalized.get("active_row_deltas", [])]
        )
    return normalized


def _build_runtime_compatibility(data: Dict[str, Any]) -> Dict[str, Any]:
    format_version = int(data.get("format_version", 1) or 1)
    capabilities = sorted(str(item) for item in data.get("format_capabilities", []))
    unsupported = [
        capability
        for capability in capabilities
        if capability not in SUPPORTED_EDGE_CAPABILITIES
    ]
    return {
        "supported": format_version <= SUPPORTED_EDGE_FORMAT_VERSION and not unsupported,
        "format_version": format_version,
        "runtime_supported_format_version": SUPPORTED_EDGE_FORMAT_VERSION,
        "capabilities": capabilities,
        "unsupported_capabilities": unsupported,
    }


def _build_manifest_digest(data: Dict[str, Any]) -> str:
    payload = {
        "format_version": data.get("format_version", 1),
        "format_capabilities": data.get("format_capabilities", []),
        "context_length": data.get("context_length", 0),
        "embed_dim": data.get("embed_dim", 0),
        "total_readout_size": data.get("total_readout_size", 0),
        "readout_storage": data.get("readout_storage", ""),
        "readout_synapses": data.get("readout_synapses", []),
        "edge_quantization": data.get("edge_quantization", {}),
        "multilevel_weight_profile": data.get("multilevel_weight_profile", {}),
        "sparse_routing_table": data.get("sparse_routing_table", {}),
        "event_compression": data.get("event_compression", {}),
        "edge_storage_profile": data.get("edge_storage_profile", {}),
        "delta_associative_state": data.get("delta_associative_state", {}),
        "spike_event_ir": data.get("spike_event_ir", {}),
        "neuromorphic_capabilities": data.get("neuromorphic_capabilities", {}),
        "neuromorphic_profile_report": data.get("neuromorphic_profile_report", {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest(data: Dict[str, Any]) -> Dict[str, Any]:
    manifest = data.get("edge_manifest", {})
    expected_digest = str(manifest.get("payload_digest", ""))
    actual_digest = _build_manifest_digest(data)
    return {
        "present": bool(manifest),
        "schema": manifest.get("schema", ""),
        "digest_algorithm": manifest.get("digest_algorithm", ""),
        "valid": bool(manifest) and expected_digest == actual_digest,
        "expected_digest": expected_digest,
        "actual_digest": actual_digest,
    }


def validate_edge_model_file(model_path: str, strict_format: bool = False) -> Dict[str, Any]:
    with open(model_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    compatibility = _build_runtime_compatibility(data)
    manifest_integrity = _validate_manifest(data)
    storage_profile = data.get("edge_storage_profile", {})
    multilevel_profile = data.get("multilevel_weight_profile", {})
    sparse_routing = _normalize_sparse_routing_table(data.get("sparse_routing_table", {}))
    delta_state = data.get("delta_associative_state", {})
    spike_event_ir = data.get("spike_event_ir", {})
    neuromorphic_capabilities = data.get("neuromorphic_capabilities", {})
    neuromorphic_profile_report = data.get("neuromorphic_profile_report", {})
    errors: List[str] = []

    if not compatibility.get("supported", False):
        unsupported = compatibility.get("unsupported_capabilities", [])
        errors.append(f"unsupported_capabilities={','.join(unsupported)}")
    if not manifest_integrity.get("valid", False):
        errors.append("manifest_integrity_failed")
    if strict_format and int(data.get("format_version", 1) or 1) > SUPPORTED_EDGE_FORMAT_VERSION:
        errors.append("format_version_too_new")
    if isinstance(delta_state, dict) and delta_state.get("enabled", False):
        if not bool(delta_state.get("budget_ok", False)):
            errors.append("delta_associative_state_budget_failed")
        if int(delta_state.get("entry_count", 0) or 0) != len(delta_state.get("entries", [])):
            errors.append("delta_associative_state_entry_count_mismatch")
    if isinstance(spike_event_ir, dict) and spike_event_ir.get("enabled", False):
        if spike_event_ir.get("schema") != "sara-spike-event-ir-v1":
            errors.append("neuromorphic_ir_schema_invalid")
        if int(spike_event_ir.get("event_count", 0) or 0) != len(spike_event_ir.get("events", [])):
            errors.append("neuromorphic_ir_event_count_mismatch")
        if not bool(spike_event_ir.get("budget_ok", False)):
            errors.append("neuromorphic_sparse_event_budget_failed")
    if isinstance(neuromorphic_capabilities, dict) and neuromorphic_capabilities.get("enabled", False):
        compatibility_map = neuromorphic_capabilities.get("backend_compatibility", {})
        profiles = neuromorphic_capabilities.get("profiles", [])
        if not isinstance(compatibility_map, dict) or not isinstance(profiles, list):
            errors.append("neuromorphic_backend_profile_invalid")
        elif not profiles or not all(bool(compatibility_map.get(str(profile), False)) for profile in profiles):
            errors.append("neuromorphic_backend_profile_incompatible")
    if isinstance(neuromorphic_profile_report, dict) and neuromorphic_profile_report.get("enabled", False):
        if neuromorphic_profile_report.get("schema") != "sara-neuromorphic-profile-report-v1":
            errors.append("neuromorphic_profile_report_schema_invalid")
        profiles = neuromorphic_profile_report.get("profiles", {})
        if not isinstance(profiles, dict):
            errors.append("neuromorphic_profile_report_invalid")
        elif int(neuromorphic_profile_report.get("profile_count", 0) or 0) != len(profiles):
            errors.append("neuromorphic_profile_report_count_mismatch")
        elif not bool(neuromorphic_profile_report.get("all_profiles_compatible", False)):
            errors.append("neuromorphic_profile_report_incompatible")

    return {
        "passed": not errors,
        "strict_format": bool(strict_format),
        "errors": errors,
        "format_version": int(data.get("format_version", 1) or 1),
        "format_capabilities": sorted(str(item) for item in data.get("format_capabilities", [])),
        "runtime_compatibility": compatibility,
        "manifest_integrity": manifest_integrity,
        "readout_storage": data.get("readout_storage", "dense_rows"),
        "row_count": int(storage_profile.get("row_count", sparse_routing.get("row_count", 0)) or 0),
        "stored_row_count": int(
            storage_profile.get("stored_row_count", sparse_routing.get("stored_row_count", 0)) or 0
        ),
        "row_reduction_ratio": float(storage_profile.get("row_reduction_ratio", 0.0) or 0.0),
        "multilevel_weight_levels": int(multilevel_profile.get("levels", 0) or 0),
        "active_row_count": len(sparse_routing.get("active_rows", [])),
        "delta_state_enabled": bool(
            isinstance(delta_state, dict) and delta_state.get("enabled", False)
        ),
        "delta_state_units": int(delta_state.get("state_units", 0) or 0)
        if isinstance(delta_state, dict)
        else 0,
        "delta_state_budget_ok": bool(delta_state.get("budget_ok", True))
        if isinstance(delta_state, dict)
        else True,
        "spike_event_ir_enabled": bool(
            isinstance(spike_event_ir, dict) and spike_event_ir.get("enabled", False)
        ),
        "spike_event_ir_schema": str(spike_event_ir.get("schema", ""))
        if isinstance(spike_event_ir, dict)
        else "",
        "spike_event_count": int(spike_event_ir.get("event_count", 0) or 0)
        if isinstance(spike_event_ir, dict)
        else 0,
        "neuromorphic_capabilities_enabled": bool(
            isinstance(neuromorphic_capabilities, dict)
            and neuromorphic_capabilities.get("enabled", False)
        ),
        "neuromorphic_profiles": [
            str(profile)
            for profile in neuromorphic_capabilities.get("profiles", [])
        ]
        if isinstance(neuromorphic_capabilities, dict)
        and isinstance(neuromorphic_capabilities.get("profiles", []), list)
        else [],
        "neuromorphic_backend_compatible": bool(
            isinstance(neuromorphic_capabilities, dict)
            and neuromorphic_capabilities.get("backend_compatibility", {})
            and all(
                bool(neuromorphic_capabilities.get("backend_compatibility", {}).get(str(profile), False))
                for profile in neuromorphic_capabilities.get("profiles", [])
            )
        ),
        "neuromorphic_profile_report_enabled": bool(
            isinstance(neuromorphic_profile_report, dict)
            and neuromorphic_profile_report.get("enabled", False)
        ),
        "neuromorphic_profile_report_schema": str(
            neuromorphic_profile_report.get("schema", "")
        )
        if isinstance(neuromorphic_profile_report, dict)
        else "",
        "neuromorphic_profile_count": int(
            neuromorphic_profile_report.get("profile_count", 0) or 0
        )
        if isinstance(neuromorphic_profile_report, dict)
        else 0,
        "neuromorphic_profile_compatibility": {
            str(profile_name): bool(profile_report.get("compatible", False))
            for profile_name, profile_report in neuromorphic_profile_report.get("profiles", {}).items()
            if isinstance(profile_report, dict)
        }
        if isinstance(neuromorphic_profile_report, dict)
        and isinstance(neuromorphic_profile_report.get("profiles", {}), dict)
        else {},
    }


class SaraEdgeRuntime:
    """
    Ultra-lightweight inference runtime for Edge devices (Raspberry Pi, Microcontrollers).
    Runs without the heavy nn.Module architecture and backpropagation structures.
    """
    def __init__(
        self,
        model_path: str,
        use_stochastic_readout: bool = False,
        stochastic_bit_count: int = 64,
        stochastic_seed: int = 7,
        strict_format: bool = False,
    ):
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.format_version = int(data.get("format_version", 1) or 1)
        self.format_capabilities = sorted(str(item) for item in data.get("format_capabilities", []))
        self.runtime_compatibility = _build_runtime_compatibility(data)
        self.edge_manifest = data.get("edge_manifest", {})
        self.manifest_integrity = _validate_manifest(data)
        if strict_format:
            if not self.runtime_compatibility.get("supported", False):
                unsupported = ", ".join(self.runtime_compatibility.get("unsupported_capabilities", []))
                raise ValueError(f"Unsupported Sara-Edge format capabilities: {unsupported}")
            if not self.manifest_integrity.get("valid", False):
                raise ValueError("Sara-Edge manifest integrity check failed.")
            delta_state = data.get("delta_associative_state", {})
            if isinstance(delta_state, dict) and delta_state.get("enabled", False):
                if not bool(delta_state.get("budget_ok", False)):
                    raise ValueError("Sara-Edge delta associative state budget check failed.")
                if int(delta_state.get("entry_count", 0) or 0) != len(delta_state.get("entries", [])):
                    raise ValueError("Sara-Edge delta associative state entry count check failed.")
            spike_event_ir = data.get("spike_event_ir", {})
            if isinstance(spike_event_ir, dict) and spike_event_ir.get("enabled", False):
                if spike_event_ir.get("schema") != "sara-spike-event-ir-v1":
                    raise ValueError("Sara-Edge neuromorphic spike event IR schema check failed.")
                if int(spike_event_ir.get("event_count", 0) or 0) != len(spike_event_ir.get("events", [])):
                    raise ValueError("Sara-Edge neuromorphic spike event IR count check failed.")
                if not bool(spike_event_ir.get("budget_ok", False)):
                    raise ValueError("Sara-Edge neuromorphic sparse event budget check failed.")
            neuromorphic_capabilities = data.get("neuromorphic_capabilities", {})
            if isinstance(neuromorphic_capabilities, dict) and neuromorphic_capabilities.get("enabled", False):
                compatibility_map = neuromorphic_capabilities.get("backend_compatibility", {})
                profiles = neuromorphic_capabilities.get("profiles", [])
                if not isinstance(compatibility_map, dict) or not isinstance(profiles, list):
                    raise ValueError("Sara-Edge neuromorphic backend profile check failed.")
                if not profiles or not all(bool(compatibility_map.get(str(profile), False)) for profile in profiles):
                    raise ValueError("Sara-Edge neuromorphic backend compatibility check failed.")
            neuromorphic_profile_report = data.get("neuromorphic_profile_report", {})
            if isinstance(neuromorphic_profile_report, dict) and neuromorphic_profile_report.get("enabled", False):
                profiles = neuromorphic_profile_report.get("profiles", {})
                if neuromorphic_profile_report.get("schema") != "sara-neuromorphic-profile-report-v1":
                    raise ValueError("Sara-Edge neuromorphic profile report schema check failed.")
                if not isinstance(profiles, dict):
                    raise ValueError("Sara-Edge neuromorphic profile report check failed.")
                if int(neuromorphic_profile_report.get("profile_count", 0) or 0) != len(profiles):
                    raise ValueError("Sara-Edge neuromorphic profile report count check failed.")
                if not bool(neuromorphic_profile_report.get("all_profiles_compatible", False)):
                    raise ValueError("Sara-Edge neuromorphic profile compatibility check failed.")
        self.context_length = data.get("context_length", 64)
        self.embed_dim = data.get("embed_dim", 64)
        self.total_readout_size = data.get("total_readout_size", 8192 + 64)
        self.readout_storage = data.get("readout_storage", "dense_rows")
        self.edge_quantization = data.get("edge_quantization", {})
        self.multilevel_weight_profile = data.get("multilevel_weight_profile", {})
        self.event_compression = data.get("event_compression", {})
        self.edge_storage_profile = data.get("edge_storage_profile", {})
        self.delta_associative_state = data.get("delta_associative_state", {})
        self.spike_event_ir = data.get("spike_event_ir", {})
        self.neuromorphic_capabilities = data.get("neuromorphic_capabilities", {})
        self.neuromorphic_profile_report = data.get("neuromorphic_profile_report", {})
        self.sparse_routing_table = _normalize_sparse_routing_table(
            data.get("sparse_routing_table", {})
        )
        
        decoded_rows: List[Dict[int, float]] = []
        for syn_dict in data.get("readout_synapses", []):
            if isinstance(syn_dict, dict) and "keys" in syn_dict and "qweights" in syn_dict:
                converted = _decode_compact_quantized_row(
                    syn_dict,
                    int(self.edge_quantization.get("bits", 0) or 0),
                )
            else:
                converted = {int(k): float(v) for k, v in syn_dict.items()}
            decoded_rows.append(converted)

        if self.readout_storage == "active_rows":
            row_count = int(self.sparse_routing_table.get("row_count", self.total_readout_size) or 0)
            active_rows = [
                int(row_index)
                for row_index in self.sparse_routing_table.get("active_rows", [])
            ]
            self.readout_synapses = [{} for _ in range(max(row_count, 0))]
            for row_index, decoded_row in zip(active_rows, decoded_rows):
                if 0 <= row_index < len(self.readout_synapses):
                    self.readout_synapses[row_index] = decoded_row
        else:
            self.readout_synapses = decoded_rows
            
        self.reservoir_size = self.total_readout_size - self.embed_dim
        self.delay_buffer: List[int] = []
        self.use_stochastic_readout = bool(use_stochastic_readout)
        self.stochastic_accumulator = StochasticAccumulator(
            bit_count=stochastic_bit_count,
            seed=stochastic_seed,
        )

    def reset_state(self) -> None:
        self.delay_buffer.clear()

    def _get_sdr(self, delay: int, tok: int) -> List[int]:
        seed_val = (delay * 73856093) ^ (tok * 19349663) ^ 42
        random.seed(seed_val)
        spikes = random.sample(range(self.reservoir_size), 20)
        random.seed()
        return spikes

    # mypy対応: Optionalを追加
    def forward_step(
        self,
        token_id: int,
        refractory_tokens: Optional[List[int]] = None,
        use_stochastic_readout: Optional[bool] = None,
    ) -> int:
        self.delay_buffer.insert(0, token_id)
        if len(self.delay_buffer) > self.context_length:
            self.delay_buffer.pop()

        res_spikes = set()
        for delay, tok in enumerate(self.delay_buffer):
            res_spikes.update(self._get_sdr(delay, tok))
        
        out_potentials: Dict[int, float] = {}
        for s in res_spikes:
            if s < len(self.readout_synapses):
                for v_idx, w in self.readout_synapses[s].items():
                    out_potentials[v_idx] = out_potentials.get(v_idx, 0.0) + w

        if refractory_tokens:
            decay_factor = 0.4
            for r_tok in reversed(refractory_tokens):
                if r_tok in out_potentials:
                    out_potentials[r_tok] *= decay_factor
                decay_factor += 0.15
                if decay_factor > 1.0:
                    decay_factor = 1.0

        if out_potentials:
            max_val = max(out_potentials.values())
            if max_val > 0.1:
                stochastic_enabled = self.use_stochastic_readout if use_stochastic_readout is None else bool(
                    use_stochastic_readout
                )
                if stochastic_enabled:
                    selected = self.stochastic_accumulator.argmax(out_potentials)
                    if selected is not None:
                        return int(selected)
                return max(out_potentials.items(), key=operator.itemgetter(1))[0]
                
        return 32 

    def generate(
        self,
        text: str,
        max_length: int = 50,
        use_stochastic_readout: Optional[bool] = None,
    ) -> str:
        input_ids = [ord(c) for c in text]
        self.reset_state()

        first_pred = 32
        for token_id in input_ids:
            first_pred = self.forward_step(token_id, use_stochastic_readout=use_stochastic_readout)

        generated_chars = []
        current_token = first_pred
        refractory_buffer: List[int] = []

        for _ in range(max_length):
            if current_token == 0:
                break

            try:
                char = chr(current_token) if current_token >= 32 else ""
            except ValueError:
                char = ""

            generated_chars.append(char)
            refractory_buffer.append(current_token)
            if len(refractory_buffer) > 6:
                refractory_buffer.pop(0)

            current_token = self.forward_step(
                current_token,
                refractory_tokens=refractory_buffer,
                use_stochastic_readout=use_stochastic_readout,
            )

        return text + "".join(generated_chars)
