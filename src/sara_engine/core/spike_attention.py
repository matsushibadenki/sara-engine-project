# パス: src/sara_engine/core/spike_attention.py
# タイトル: スパイクベース・アテンション
# 目的: 行列演算や誤差逆伝播を使用せず、スパイクの同時発火（Coincidence）と時間的減衰を用いてTransformerのAttentionに相当する情報の重み付けを行う。

import math
import typing

class SpikeAttention:
    def __init__(self, decay_rate: float = 0.9, threshold: float = 1.0):
        self.decay_rate = decay_rate
        self.threshold = threshold
        # メモリ（過去の発火履歴）を保持
        self.history_keys = []
        self.history_values = []

    def process_step(self, query_spikes: list[int], key_spikes: list[int], value_spikes: list[int]) -> tuple[list[int], list[float]]:
        """
        1タイムステップにおけるスパイクの処理。
        query_spikes, key_spikes, value_spikes: 各ニューロンの発火状態 (0 または 1) のリスト
        """
        self.history_keys.append(key_spikes)
        self.history_values.append(value_spikes)
        
        seq_length = len(self.history_keys)
        num_neurons = len(query_spikes)
        
        output_spikes = [0] * num_neurons
        attention_scores = [0.0] * seq_length # 可視化用のスコア

        # クエリが発火していない場合は、処理をスキップしてエネルギーを節約（Event-driven）
        if sum(query_spikes) == 0:
            return output_spikes, attention_scores

        # 過去のKeyと現在のQueryの重なり合い（Overlap）を計算（行列演算不使用）
        for t in range(seq_length):
            past_keys = self.history_keys[t]
            overlap_count = 0
            
            # 各ニューロンのスパイクの一致を確認
            for i in range(num_neurons):
                if query_spikes[i] == 1 and past_keys[i] == 1:
                    overlap_count += 1
            
            # 時間的減衰（古い記憶ほど影響が少ない）
            time_diff = seq_length - 1 - t
            decay = self.decay_rate ** time_diff
            score = overlap_count * decay
            attention_scores[t] = score

        # 最もスコアの高い過去のタイミング（Winner-Take-All的アプローチ）を特定
        max_score = 0.0
        best_t = -1
        for t in range(seq_length):
            if attention_scores[t] > max_score:
                max_score = attention_scores[t]
                best_t = t

        # 閾値を超えた場合に、その時点のValueを元に発火を生成
        if max_score >= self.threshold and best_t != -1:
            best_values = self.history_values[best_t]
            for i in range(num_neurons):
                # 単純化のため、選ばれた過去のValueが発火していれば出力も発火
                if best_values[i] == 1:
                    output_spikes[i] = 1

        return output_spikes, attention_scores

    def reset_state(self):
        """状態の初期化"""
        self.history_keys = []
        self.history_values = []