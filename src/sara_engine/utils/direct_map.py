# Directory Path: src/sara_engine/utils/direct_map.py
# English Title: Direct Map Serialization Helpers
# Purpose/Content: Safely restores sparse SNN direct-memory maps from serialized string keys without using eval().

import ast
from typing import Dict, Mapping, Tuple


DirectMap = Dict[Tuple[int, ...], Dict[int, float]]


def restore_direct_map(raw_map: Mapping[object, Mapping[object, object]]) -> DirectMap:
    restored: DirectMap = {}
    for raw_key, raw_values in raw_map.items():
        key = _parse_direct_map_key(raw_key)
        if not isinstance(raw_values, Mapping):
            raise ValueError("direct_map values must be dictionaries.")
        restored[key] = {int(token_id): float(weight) for token_id, weight in raw_values.items()}
    return restored


def serialize_direct_map(direct_map: Mapping[Tuple[int, ...], Mapping[int, float]]) -> Dict[str, Dict[str, float]]:
    serialized: Dict[str, Dict[str, float]] = {}
    for key, values in direct_map.items():
        serialized[str(tuple(int(item) for item in key))] = {
            str(int(token_id)): float(weight) for token_id, weight in values.items()
        }
    return serialized


def _parse_direct_map_key(raw_key: object) -> Tuple[int, ...]:
    if isinstance(raw_key, tuple):
        return tuple(int(item) for item in raw_key)
    if isinstance(raw_key, list):
        return tuple(int(item) for item in raw_key)
    if not isinstance(raw_key, str):
        raise ValueError("direct_map key must be a string, list, or tuple.")

    parsed = ast.literal_eval(raw_key)
    if isinstance(parsed, tuple):
        return tuple(int(item) for item in parsed)
    if isinstance(parsed, list):
        return tuple(int(item) for item in parsed)
    raise ValueError("direct_map key must decode to a tuple or list of integers.")
