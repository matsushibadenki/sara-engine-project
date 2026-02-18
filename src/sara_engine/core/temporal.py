_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/temporal.py",
    "//": "タイトル: Temporal Coding (位置表現) レイヤー",
    "//": "目的: 行列演算を用いず、SDRの巡回シフトと位置アンカーを用いて、系列内の語順（構文情報）を表現する。"
}

import random
from typing import List, Set

class TemporalEncoder:
    """
    SNN向けの位置エンコーディング。
    標準リストと集合演算で実装。
    """
    def __init__(self, input_size: int, target_density: float = 0.05):
        self.input_size = input_size
        self.target_density = target_density
        self.position_anchors: dict[int, List[int]] = {}
        self.shift_base = 31 

    def _get_position_anchor(self, position: int) -> List[int]:
        if position in self.position_anchors:
            return self.position_anchors[position]
        
        # 決定論的な乱数生成で位置固有のSDRを作成
        rng = random.Random(position * 9999 + 1234)
        num_active = max(1, int(self.input_size * 0.02)) # 薄い密度
        anchor = sorted(rng.sample(range(self.input_size), num_active))
        
        self.position_anchors[position] = anchor
        return anchor

    def encode(self, sdr: List[int], position: int) -> List[int]:
        """
        入力SDRに対して位置情報を付与する。
        1. 巡回シフト (位相)
        2. 位置アンカーとの和集合 (振幅)
        """
        # 1. Cyclic Shift
        shift_amount = (position * self.shift_base) % self.input_size
        shifted_spikes = set()
        for idx in sdr:
            new_idx = (idx + shift_amount) % self.input_size
            shifted_spikes.add(new_idx)

        # 2. Additive Positional Binding
        anchor_spikes = self._get_position_anchor(position)
        bound_spikes = shifted_spikes.union(set(anchor_spikes))

        # 3. Sparsity Maintenance
        target_len = max(1, int(self.input_size * self.target_density))
        bound_list = sorted(list(bound_spikes))
        
        if len(bound_list) > target_len:
            # 決定論的サンプリング
            seed_val = sum(bound_list[:5]) + position
            rng = random.Random(seed_val)
            return sorted(rng.sample(bound_list, target_len))
        
        return bound_list

    def decode_sequence(self, sequence_sdrs: List[List[int]]) -> List[List[int]]:
        temporal_sequence = []
        for t, sdr in enumerate(sequence_sdrs):
            temporal_sequence.append(self.encode(sdr, t))
        return temporal_sequence