_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/encoders/physical.py",
    "//": "タイトル: 物理状態(NeuroFEM)用 SDRエンコーダー",
    "//": "目的: NeuroFEMで得られた連続値の物理状態（温度や変位など）を、SARA大脳皮質が処理できるSDR（スパース分散表現）に変換する。"
}

import numpy as np
from typing import List

class PhysicalScalarEncoder:
    """
    連続値（スカラー）をSDRに変換するエンコーダー。
    HTMのScalar Encoderの概念を応用し、近い値は重複するビット列（トポロジーの保存）を持つようにエンコードする。
    """
    def __init__(self, sdr_size: int, min_val: float, max_val: float, active_bits: int):
        self.sdr_size = sdr_size
        self.min_val = min_val
        self.max_val = max_val
        self.active_bits = active_bits
        
        # 表現可能なバケット（段階）の数
        self.num_buckets = self.sdr_size - self.active_bits + 1

    def encode(self, value: float) -> List[int]:
        # 値を範囲内にクリップ
        clipped_value = max(self.min_val, min(value, self.max_val))
        
        # 値を0〜1に正規化
        if self.max_val == self.min_val:
            normalized = 0.5
        else:
            normalized = (clipped_value - self.min_val) / (self.max_val - self.min_val)
            
        # どのバケットに属するかを計算 (行列演算を使わず算出)
        bucket_idx = int(np.floor(normalized * (self.num_buckets - 1)))
        
        # 連続した active_bits 個のインデックスを発火させる（トポロジーの保存）
        sdr_indices = list(range(bucket_idx, bucket_idx + self.active_bits))
        return sdr_indices

class GridStateEncoder:
    """
    NeuroFEMの2D/1Dグリッド全体の状態を結合し、1つの巨大な空間的SDRとして表現する。
    """
    def __init__(self, num_nodes: int, node_sdr_size: int = 64, min_val: float = 0.0, max_val: float = 30.0, active_bits: int = 4):
        self.num_nodes = num_nodes
        self.node_sdr_size = node_sdr_size
        self.scalar_encoder = PhysicalScalarEncoder(node_sdr_size, min_val, max_val, active_bits)
        self.total_sdr_size = num_nodes * node_sdr_size

    def encode_grid(self, grid_state: List[float]) -> List[int]:
        combined_sdr = []
        for i, value in enumerate(grid_state):
            # 各ノードの値をSDR化
            node_sdr = self.scalar_encoder.encode(value)
            
            # グローバルなインデックス空間にシフトして結合
            shifted_sdr = [idx + (i * self.node_sdr_size) for idx in node_sdr]
            combined_sdr.extend(shifted_sdr)
            
        return combined_sdr