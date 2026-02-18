_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/spike_attention.py",
    "//": "タイトル: Spike-based Attention (Standalone)",
    "//": "目的: 行列演算や誤差逆伝播を使用せず、スパイクの同時発火（Coincidence）と時間的減衰を用いてAttentionを行う単体モジュール。"
}

import math
from typing import List, Tuple

class SpikeAttention:
    def __init__(self, decay_rate: float = 0.9, threshold: float = 1.0):
        self.decay_rate = decay_rate
        self.threshold = threshold
        # メモリ（過去の発火履歴）を保持: List[List[int]] (各ステップのスパイクベクトル)
        # 注意: ここでの入力はインデックスリストではなく、0/1のリストを想定している古いIFの場合があるが、
        # プロジェクト全体の方針に合わせてインデックスリスト(SDR)を扱うように調整する。
        # ただし、既存コードとの互換性のため、呼び出し元が0/1配列を渡してくるならそれに対応する必要がある。
        # ここでは「インデックスのリスト」を扱うモダンなSARA形式とする。
        self.history_keys: List[List[int]] = [] # 各ステップの発火インデックスリスト
        self.history_values: List[List[int]] = []

    def process_step(self, query_spikes: List[int], key_spikes: List[int], value_spikes: List[int]) -> Tuple[List[int], List[float]]:
        """
        1タイムステップにおけるスパイクの処理。
        Args:
            query_spikes: 発火したニューロンのインデックスリスト
            key_spikes: 発火したニューロンのインデックスリスト
            value_spikes: 発火したニューロンのインデックスリスト
        """
        self.history_keys.append(key_spikes)
        self.history_values.append(value_spikes)
        
        seq_length = len(self.history_keys)
        
        output_spikes: List[int] = []
        attention_scores: List[float] = [0.0] * seq_length # 可視化用のスコア

        # クエリが空の場合はスキップ（省エネ）
        if not query_spikes:
            return [], attention_scores

        q_set = set(query_spikes)

        # 過去のKeyと現在のQueryの重なり合い（Overlap）を計算
        for t in range(seq_length):
            past_keys = self.history_keys[t]
            # 集合積のサイズ
            overlap_count = 0
            for k in past_keys:
                if k in q_set:
                    overlap_count += 1
            
            # 時間的減衰
            time_diff = seq_length - 1 - t
            decay = self.decay_rate ** time_diff
            score = overlap_count * decay
            attention_scores[t] = score

        # 最もスコアの高い過去のタイミング（Winner-Take-All）
        max_score = 0.0
        best_t = -1
        for t in range(seq_length):
            if attention_scores[t] > max_score:
                max_score = attention_scores[t]
                best_t = t

        # 閾値を超えた場合に、その時点のValueを出力
        if max_score >= self.threshold and best_t != -1:
            output_spikes = self.history_values[best_t]

        return output_spikes, attention_scores

    def reset(self):
        self.history_keys = []
        self.history_values = []