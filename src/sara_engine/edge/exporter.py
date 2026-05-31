# ディレクトリパス: src/sara_engine/edge/exporter.py
# ファイルの日本語タイトル: エッジ用モデルエクスポーター
# ファイルの目的や内容: 学習済みのSARAモデルからシナプス重みだけを抽出し、エッジデバイスで読み込める軽量なフォーマットにシリアライズする。
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional

from .neuromorphic import (
    NeuromorphicProfileInput,
    build_neuromorphic_capabilities,
    build_neuromorphic_profile_report,
    build_spike_event_ir,
    normalize_neuromorphic_profiles,
)

EDGE_FORMAT_VERSION = 2


def _extract_weight(value: Any) -> float:
    if isinstance(value, tuple) and value:
        return float(value[0])
    if isinstance(value, list) and value:
        return float(value[0])
    return float(value)


def _normalize_readout_row(row: Mapping[Any, Any]) -> Dict[int, float]:
    normalized: Dict[int, float] = {}
    for post_id, raw_weight in row.items():
        normalized[int(post_id)] = _extract_weight(raw_weight)
    return normalized


def _quantize_row(row: Dict[int, float], quantization_bits: int) -> Dict[int, float]:
    if not row:
        return {}
    if quantization_bits < 2:
        return dict(row)

    levels = max(2, 2 ** int(quantization_bits))
    values = list(row.values())
    min_val = min(values)
    max_val = max(values)
    if max_val <= min_val:
        return dict(row)

    quantized: Dict[int, float] = {}
    scale = (levels - 1) / (max_val - min_val)
    for post_id, weight in row.items():
        q_value = int(round((weight - min_val) * scale))
        q_value = max(0, min(levels - 1, q_value))
        quantized[post_id] = min_val + (q_value / (levels - 1)) * (max_val - min_val)
    return quantized


def _compact_quantize_row(row: Dict[int, float], quantization_bits: int) -> Dict[str, Any]:
    if not row:
        return {"keys": [], "qweights": [], "min": 0.0, "max": 0.0}

    levels = max(2, 2 ** int(quantization_bits))
    items = sorted(row.items())
    values = [float(weight) for _, weight in items]
    min_val = min(values)
    max_val = max(values)
    if max_val <= min_val:
        return {
            "keys": [int(post_id) for post_id, _ in items],
            "qweights": [0 for _ in items],
            "min": min_val,
            "max": max_val,
        }

    scale = (levels - 1) / (max_val - min_val)
    qweights: List[int] = []
    for _, weight in items:
        q_value = int(round((float(weight) - min_val) * scale))
        qweights.append(max(0, min(levels - 1, q_value)))
    return {
        "keys": [int(post_id) for post_id, _ in items],
        "qweights": qweights,
        "min": min_val,
        "max": max_val,
    }


def _delta_encode_ids(ids: List[int]) -> List[int]:
    if not ids:
        return []

    encoded: List[int] = []
    previous = 0
    for index, value in enumerate(sorted(set(int(item) for item in ids))):
        if index == 0:
            encoded.append(value)
        else:
            encoded.append(value - previous)
        previous = value
    return encoded


def _build_capabilities(
    quantization_bits: Optional[int],
    compact_quantized: bool,
    compress_events: bool,
    sparse_readout: bool,
    delta_associative_state: Optional[Mapping[str, Any]] = None,
    neuromorphic_profile: NeuromorphicProfileInput = None,
    neuromorphic_state_traces: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[str]:
    capabilities = ["dense_row_index_space", "sparse_routing_table"]
    if not sparse_readout:
        capabilities.append("dense_readout_rows")
    if quantization_bits is not None:
        capabilities.append("multilevel_weights")
        if compact_quantized:
            capabilities.append("compact_int_weights")
        else:
            capabilities.append("float_quantized_weights")
    if compress_events:
        capabilities.append("delta_event_compression")
    if sparse_readout:
        capabilities.append("active_row_readout_storage")
    if delta_associative_state is not None:
        capabilities.append("delta_associative_memory_state")
    if normalize_neuromorphic_profiles(neuromorphic_profile):
        capabilities.append("neuromorphic_spike_event_ir")
        capabilities.append("neuromorphic_backend_profile")
    if neuromorphic_state_traces:
        capabilities.append("neuromorphic_state_trace_ir")
    return sorted(capabilities)


def _normalize_delta_associative_state(state: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if state is None:
        return {
            "enabled": False,
            "storage": "none",
            "capacity": 0,
            "state_units": 0,
            "entry_count": 0,
            "budget_ok": True,
            "entries": [],
        }

    entries: List[Dict[str, Any]] = []
    for raw_entry in state.get("entries", []):
        if not isinstance(raw_entry, Mapping):
            continue
        entries.append(
            {
                "context_id": int(raw_entry.get("context_id", 0) or 0),
                "value_id": int(raw_entry.get("value_id", 0) or 0),
                "weight": float(raw_entry.get("weight", 0.0) or 0.0),
                "last_update": int(raw_entry.get("last_update", 0) or 0),
            }
        )
    entries = sorted(entries, key=lambda entry: (entry["context_id"], entry["value_id"]))
    capacity = int(state.get("capacity", len(entries)) or len(entries))
    state_units = int(state.get("state_units", len(entries)) or len(entries))
    return {
        "enabled": True,
        "storage": "compact_sparse_float",
        "capacity": max(0, capacity),
        "state_units": max(0, state_units),
        "entry_count": len(entries),
        "budget_ok": state_units <= max(0, capacity),
        "entries": entries,
    }




def _build_manifest_digest(edge_data: Dict[str, Any]) -> str:
    payload = {
        "format_version": edge_data.get("format_version", 1),
        "format_capabilities": edge_data.get("format_capabilities", []),
        "context_length": edge_data.get("context_length", 0),
        "embed_dim": edge_data.get("embed_dim", 0),
        "total_readout_size": edge_data.get("total_readout_size", 0),
        "readout_storage": edge_data.get("readout_storage", ""),
        "readout_synapses": edge_data.get("readout_synapses", []),
        "edge_quantization": edge_data.get("edge_quantization", {}),
        "multilevel_weight_profile": edge_data.get("multilevel_weight_profile", {}),
        "sparse_routing_table": edge_data.get("sparse_routing_table", {}),
        "event_compression": edge_data.get("event_compression", {}),
        "edge_storage_profile": edge_data.get("edge_storage_profile", {}),
        "delta_associative_state": edge_data.get("delta_associative_state", {}),
        "spike_event_ir": edge_data.get("spike_event_ir", {}),
        "neuromorphic_capabilities": edge_data.get("neuromorphic_capabilities", {}),
        "neuromorphic_profile_report": edge_data.get("neuromorphic_profile_report", {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _attach_edge_manifest(edge_data: Dict[str, Any]) -> None:
    storage_profile = edge_data.get("edge_storage_profile", {})
    edge_data["edge_manifest"] = {
        "schema": "sara-edge-manifest-v1",
        "digest_algorithm": "sha256",
        "payload_digest": _build_manifest_digest(edge_data),
        "format_version": edge_data.get("format_version", 1),
        "capability_count": len(edge_data.get("format_capabilities", [])),
        "readout_storage": edge_data.get("readout_storage", ""),
        "stored_row_count": int(storage_profile.get("stored_row_count", 0) or 0),
        "synapse_count": int(storage_profile.get("synapse_count", 0) or 0),
    }


def export_for_edge(
    model: Any,
    filepath: str,
    quantization_bits: Optional[int] = None,
    compact_quantized: bool = False,
    compress_events: bool = False,
    sparse_readout: bool = False,
    delta_associative_state: Optional[Mapping[str, Any]] = None,
    neuromorphic_profile: NeuromorphicProfileInput = None,
    neuromorphic_state_traces: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    """
    Extracts essential inference parameters (e.g., readout synapses, context length)
    from a SpikingTransformerModel and saves it as a lightweight JSON for Sara-Edge.
    """
    # mypy対応: 辞書の型を明示
    edge_data: Dict[str, Any] = {
        "format_version": EDGE_FORMAT_VERSION,
        "format_capabilities": _build_capabilities(
            quantization_bits,
            compact_quantized,
            compress_events,
            sparse_readout,
            delta_associative_state,
            neuromorphic_profile,
            neuromorphic_state_traces,
        ),
        "context_length": getattr(model, "context_length", 64),
        "embed_dim": getattr(model.config, "embed_dim", 64) if hasattr(model, "config") else 64,
        "total_readout_size": getattr(model, "total_readout_size", 8192 + 64),
        "readout_synapses": [],
        "readout_storage": "active_rows" if sparse_readout else "dense_rows",
        "edge_quantization": {
            "enabled": bool(quantization_bits is not None),
            "bits": int(quantization_bits or 0),
            "storage": "compact_int" if compact_quantized and quantization_bits is not None else "float",
        },
        "multilevel_weight_profile": {
            "enabled": False,
            "bits": 0,
            "levels": 0,
            "storage": "none",
            "quantized_row_count": 0,
            "quantized_weight_count": 0,
            "flat_row_count": 0,
        },
        "sparse_routing_table": {
            "enabled": True,
            "active_rows": [],
            "row_count": 0,
        },
        "event_compression": {
            "enabled": False,
            "encoding": "none",
            "uncompressed_event_count": 0,
            "compressed_event_count": 0,
        },
        "edge_storage_profile": {
            "enabled": True,
            "row_count": 0,
            "stored_row_count": 0,
            "active_row_count": 0,
            "empty_row_count": 0,
            "row_reduction_ratio": 0.0,
            "synapse_count": 0,
            "compact_weight_count": 0,
        },
        "delta_associative_state": _normalize_delta_associative_state(delta_associative_state),
        "spike_event_ir": {
            "enabled": False,
            "schema": "none",
            "event_count": 0,
            "budget_ok": True,
            "events": [],
        },
        "neuromorphic_capabilities": {
            "enabled": False,
            "profiles": [],
            "backend_compatibility": {},
        },
        "neuromorphic_profile_report": {
            "enabled": False,
            "schema": "sara-neuromorphic-profile-report-v1",
            "profile_count": 0,
            "profiles": {},
            "all_profiles_compatible": True,
        },
    }
    
    # Convert readout_synapses to JSON-serializable rows of scalar weights.
    if hasattr(model, "readout_synapses"):
        active_rows: List[int] = []
        exported_rows: List[Any] = []
        synapse_count = 0
        compact_weight_count = 0
        quantized_row_count = 0
        quantized_weight_count = 0
        flat_row_count = 0
        readout_row_count = len(model.readout_synapses)
        for row_index, synapses in enumerate(model.readout_synapses):
            if not isinstance(synapses, Mapping):
                if not sparse_readout:
                    exported_rows.append({})
                continue
            normalized = _normalize_readout_row(synapses)
            synapse_count += len(normalized)
            if quantization_bits is not None:
                if normalized:
                    quantized_row_count += 1
                    quantized_weight_count += len(normalized)
                    values = list(normalized.values())
                    if values and max(values) <= min(values):
                        flat_row_count += 1
                if compact_quantized:
                    exported_row = _compact_quantize_row(normalized, quantization_bits)
                    compact_weight_count += len(exported_row.get("qweights", []))
                else:
                    exported_row = _quantize_row(normalized, quantization_bits)
            else:
                exported_row = normalized
            if normalized:
                active_rows.append(int(row_index))
                exported_rows.append(exported_row)
            elif not sparse_readout:
                exported_rows.append(exported_row)
        edge_data["readout_synapses"] = exported_rows
        active_row_count = len(active_rows)
        stored_row_count = len(exported_rows)
        row_reduction_ratio = 1.0 - (stored_row_count / max(readout_row_count, 1))
        edge_data["edge_storage_profile"] = {
            "enabled": True,
            "row_count": readout_row_count,
            "stored_row_count": stored_row_count,
            "active_row_count": active_row_count,
            "empty_row_count": max(0, readout_row_count - active_row_count),
            "row_reduction_ratio": max(0.0, row_reduction_ratio),
            "synapse_count": synapse_count,
            "compact_weight_count": compact_weight_count,
        }
        edge_data["multilevel_weight_profile"] = {
            "enabled": bool(quantization_bits is not None),
            "bits": int(quantization_bits or 0),
            "levels": max(2, 2 ** int(quantization_bits or 0)) if quantization_bits is not None else 0,
            "storage": edge_data["edge_quantization"]["storage"],
            "quantized_row_count": quantized_row_count,
            "quantized_weight_count": quantized_weight_count,
            "flat_row_count": flat_row_count,
        }
        sparse_routing_table: Dict[str, Any] = {
            "enabled": True,
            "row_count": readout_row_count,
            "stored_row_count": stored_row_count,
        }
        if compress_events:
            active_row_deltas = _delta_encode_ids(active_rows)
            sparse_routing_table.update(
                {
                    "encoding": "delta",
                    "active_row_deltas": active_row_deltas,
                }
            )
            edge_data["event_compression"] = {
                "enabled": True,
                "encoding": "delta",
                "uncompressed_event_count": len(active_rows),
                "compressed_event_count": len(active_row_deltas),
            }
        else:
            sparse_routing_table["active_rows"] = active_rows
        edge_data["sparse_routing_table"] = sparse_routing_table
        edge_data["spike_event_ir"] = build_spike_event_ir(
            active_rows=active_rows,
            context_length=int(edge_data["context_length"]),
            total_readout_size=int(edge_data["total_readout_size"]),
            quantization_bits=quantization_bits,
            compact_quantized=compact_quantized,
            compress_events=compress_events,
            delta_state=edge_data["delta_associative_state"],
            neuromorphic_profile=neuromorphic_profile,
            state_traces=neuromorphic_state_traces,
        )
        edge_data["neuromorphic_capabilities"] = build_neuromorphic_capabilities(
            spike_event_ir=edge_data["spike_event_ir"],
            delta_state=edge_data["delta_associative_state"],
            quantization_bits=quantization_bits,
            neuromorphic_profile=neuromorphic_profile,
        )
        edge_data["neuromorphic_profile_report"] = build_neuromorphic_profile_report(
            spike_event_ir=edge_data["spike_event_ir"],
            neuromorphic_capabilities=edge_data["neuromorphic_capabilities"],
        )

    _attach_edge_manifest(edge_data)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(edge_data, f)
    
    print(f"Model successfully exported for Sara-Edge at: {filepath}")
