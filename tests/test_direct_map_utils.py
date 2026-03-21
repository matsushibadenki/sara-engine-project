import os
import sys
from typing import Mapping, cast

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.utils.direct_map import restore_direct_map, serialize_direct_map


def test_restore_direct_map_accepts_tuple_and_list_keys():
    restored = restore_direct_map(
        {
            "(1, 2, 3)": {"10": 0.5},
            "[4, 5]": {"11": 1.25},
        }
    )

    assert restored[(1, 2, 3)][10] == 0.5
    assert restored[(4, 5)][11] == 1.25


def test_restore_direct_map_rejects_unsafe_key_expression():
    with pytest.raises((ValueError, SyntaxError)):
        restore_direct_map({"__import__('os').system('echo bad')": {"1": 1.0}})


def test_restore_direct_map_rejects_non_mapping_values():
    raw_map = cast(Mapping[object, Mapping[object, object]], {"(1, 2)": [1, 2, 3]})
    with pytest.raises(ValueError, match="must be dictionaries"):
        restore_direct_map(raw_map)


def test_serialize_direct_map_roundtrip():
    direct_map = {
        (1, 2, 3): {10: 0.5, 11: 1.0},
        (4,): {12: 2.5},
    }

    serialized = serialize_direct_map(direct_map)
    serialized_map = cast(Mapping[object, Mapping[object, object]], serialized)
    restored = restore_direct_map(serialized_map)

    assert restored == direct_map
