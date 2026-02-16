# SARA Engine Temporal Coding Module
_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/temporal.py",
    "//": "タイトル: Temporal Coding (位置表現) レイヤー",
    "//": "目的: 行列演算を用いず、SDRの巡回シフト（位相コーディングの模倣）と位置アンカーを用いて、系列内の語順（構文情報）を表現する。"
}

import random
from typing import List, Set

class TemporalEncoder:
    """
    SNN向けの位置エンコーディング（Temporal Coding）。
    SDRのスパイクパターンに対して、発火位置のインデックスを時間ステップ（語順）に応じて
    巡回シフト（Circular Shift）させることで、位相（Phase）のズレを表現する。
    また、絶対位置を示すアンカースパイクを結合することで、TransformerのPositional Encodingに
    相当する機能を誤差逆伝播法や行列演算なしで実現する。
    """
    def __init__(self, input_size: int, target_density: float = 0.05):
        self.input_size = input_size
        self.target_density = target_density
        self.position_anchors: dict[int, List[int]] = {}
        # 巡回シフトのステップ量（互いに素になるような素数を選ぶことでビットの衝突を防ぐ）
        self.shift_base = 31 

    def _get_position_anchor(self, position: int) -> List[int]:
        """
        特定の位置（0, 1, 2...）に対して、固定のスパイクパターン（絶対位置アンカー）を決定論的に生成する。
        """
        if position not in self.position_anchors:
            # 位置ごとに一意なシードを用いてランダムなスパイクを生成
            rng = random.Random(12345 + position)
            target_w = max(1, int(self.input_size * self.target_density))
            anchor = rng.sample(range(self.input_size), target_w)
            self.position_anchors[position] = sorted(anchor)
        return self.position_anchors[position]

    def encode(self, sdr: List[int], position: int) -> List[int]:
        """
        入力SDRに位置情報（Temporal Context）を埋め込む。
        """
        if not sdr:
            return []

        # 1. 位相シフト（Phase Shift / Circular Shift）
        # 生物学的なスパイクタイミングのズレ（Axonal Delay）を、
        # 空間的な発火インデックスのシフトとして模倣する。
        shift_amount = (position * self.shift_base) % self.input_size
        
        shifted_spikes: Set[int] = set()
        for neuron_id in sdr:
            new_id = (neuron_id + shift_amount) % self.input_size
            shifted_spikes.add(new_id)

        # 2. 絶対位置アンカーの結合 (Additive Positional Binding)
        # 語順の絶対的な位置を示すSDRを重ね合わせる（Set Union）
        anchor_spikes = self._get_position_anchor(position)
        bound_spikes = shifted_spikes.union(set(anchor_spikes))

        # 3. スパース性の維持 (Homeostasis / Thresholding)
        # 重ね合わせによってスパイクが増えすぎた場合、ネットワーク全体の発火密度を維持するために間引く。
        # 単純なランダムサンプリングではなく、入力SDRのシグネチャを残すために決定論的にダウンサンプリングする。
        target_len = max(1, int(self.input_size * self.target_density))
        bound_list = sorted(list(bound_spikes))
        
        if len(bound_list) > target_len:
            # 安定した間引きを行うために、発火パターンの特徴をシードとして使う
            seed_val = sum(bound_list[:5]) + position
            rng = random.Random(seed_val)
            return sorted(rng.sample(bound_list, target_len))
        
        return bound_list

    def decode_sequence(self, sequence_sdrs: List[List[int]]) -> List[List[int]]:
        """
        シーケンス（文章）全体のSDRリストに対して、順番に一括で位置情報を付与する。
        """
        temporal_sequence = []
        for t, sdr in enumerate(sequence_sdrs):
            temporal_sequence.append(self.encode(sdr, t))
        return temporal_sequence

    def compute_temporal_overlap(self, sdr_a: List[int], pos_a: int, sdr_b: List[int], pos_b: int) -> int:
        """
        2つのSDRが「同じ単語で同じ位置」にある場合のみ高いOverlap（スパイクの一致度）を返すかを
        計算するためのデバッグ・検証用ユーティリティ。
        """
        encoded_a = set(self.encode(sdr_a, pos_a))
        encoded_b = set(self.encode(sdr_b, pos_b))
        return len(encoded_a.intersection(encoded_b))