_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/readout_layer.py",
    "//": "タイトル: Adam最適化機能付き読み出し層",
    "//": "目的: 行列演算を排除し、スパースなリザーバー状態からの教師あり学習（Adam）と動的枝刈り（Sleep Phase）を実装する。"
}

import math
import random
from typing import List, Dict

class ReadoutLayer:
    def __init__(self, input_size: int, output_size: int, learning_rate: float = 0.002):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = learning_rate
        self.t = 0
        
        self.beta1 = 0.93
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        # 行列演算を使わず、辞書で重みとAdamのモーメンタムを個別に管理する
        # 形式: self.weights[class_id][input_neuron_id] = weight_value
        self.weights: List[Dict[int, float]] = [{} for _ in range(output_size)]
        self.m: List[Dict[int, float]] = [{} for _ in range(output_size)]
        self.v: List[Dict[int, float]] = [{} for _ in range(output_size)]
        
        # 初期の結合（必要に応じて初期からスパースにすることも可能）
        limit = math.sqrt(2.0 / max(1, input_size))
        for o in range(output_size):
            for i in range(input_size):
                self.weights[o][i] = random.uniform(-limit, limit)
                self.m[o][i] = 0.0
                self.v[o][i] = 0.0

    def predict(self, active_hidden_indices: List[int]) -> List[float]:
        """発火した隠れ層ニューロンのインデックスから、各クラスのポテンシャルを計算"""
        potentials = [0.0] * self.output_size
        
        if not active_hidden_indices:
            return potentials

        scale_factor = 14.0 / (len(active_hidden_indices) + 12.0)
        
        # 辞書を用いたスパースな内積計算
        for o in range(self.output_size):
            s = 0.0
            for idx in active_hidden_indices:
                if idx in self.weights[o]:
                    s += self.weights[o][idx]
            potentials[o] = s * scale_factor
            
        # 平均引きによるソフト抑制
        max_p = max(potentials)
        if max_p > 0:
            mean_p = sum(potentials) / self.output_size
            for o in range(self.output_size):
                potentials[o] -= 0.05 * mean_p
                # 発散を防ぐ物理的クリッピング
                if potentials[o] < -6.0: potentials[o] = -6.0
                if potentials[o] > 6.0: potentials[o] = 6.0
                
        return potentials

    def train_step(self, active_hidden_indices: List[int], target_label: int):
        """Adam最適化を用いた1ステップの学習 (行列計算・多層BP不使用)"""
        if not active_hidden_indices:
            return
            
        self.t += 1
        current_lr = self.lr / (1.0 + 0.0015 * self.t)
        
        potentials = self.predict(active_hidden_indices)
        errors = [0.0] * self.output_size
        
        # 正解クラスは活動ポテンシャル2.4を目指す
        if potentials[target_label] < 2.4:
            errors[target_label] = 2.4 - potentials[target_label]
            
        # ソフト・ネガティブ: 不正解クラスは緩やかに-0.5へ押し戻し、過剰な回路破壊を防ぐ
        for o in range(self.output_size):
            if o != target_label and potentials[o] > 0.0:
                errors[o] = -0.5 - potentials[o]
                
        # 発火したシナプス（関連する結合）のみをAdamで局所的に更新する
        for o in range(self.output_size):
            err = errors[o]
            if abs(err) <= 0.01:
                continue
                
            for idx in active_hidden_indices:
                # 枝刈りによって結合が消滅していた場合は再生成する
                if idx not in self.weights[o]:
                    self.weights[o][idx] = 0.0
                    self.m[o][idx] = 0.0
                    self.v[o][idx] = 0.0
                
                # エラー値を勾配として使用し、モーメンタムを計算
                g = err 
                
                self.m[o][idx] = self.beta1 * self.m[o][idx] + (1 - self.beta1) * g
                self.v[o][idx] = self.beta2 * self.v[o][idx] + (1 - self.beta2) * (g * g)
                
                m_hat = self.m[o][idx] / (1 - (self.beta1 ** min(self.t, 1000)))
                v_hat = self.v[o][idx] / (1 - (self.beta2 ** min(self.t, 1000)))
                
                self.weights[o][idx] += current_lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
                
                # 重みの物理的限界を設ける
                if self.weights[o][idx] < -5.0: self.weights[o][idx] = -5.0
                if self.weights[o][idx] > 5.0: self.weights[o][idx] = 5.0

    def sleep_phase(self, prune_rate: float = 0.025) -> str:
        """睡眠フェーズ：重要度の低い結合の枝刈りと、ネットワークの全体スケーリング"""
        pruned_total = 0
        total_weights = 0
        
        for o in range(self.output_size):
            # 1. わずかな自然減衰
            for idx in list(self.weights[o].keys()):
                self.weights[o][idx] *= 0.998
                
            # 2. 枝刈り閾値の動的計算（0.0埋めを避けるため絶対値で評価）
            active_weights = [abs(w) for w in self.weights[o].values() if abs(w) > 1e-6]
            if not active_weights:
                continue
                
            active_weights.sort()
            prune_idx = int(len(active_weights) * prune_rate)
            threshold = active_weights[prune_idx] if prune_idx < len(active_weights) else 0.0
            
            # 3. 枝刈りの実行と、残存結合のノルム計算
            keys_to_remove = []
            sq_sum = 0.0
            
            for idx, w in self.weights[o].items():
                total_weights += 1
                if abs(w) < threshold:
                    keys_to_remove.append(idx)
                    pruned_total += 1
                else:
                    sq_sum += w * w
                    
            for idx in keys_to_remove:
                del self.weights[o][idx]
                if idx in self.m[o]: del self.m[o][idx]
                if idx in self.v[o]: del self.v[o][idx]
                
            # 4. ネットワーク状態を安定させるためのスケーリング
            norm = math.sqrt(sq_sum)
            target_norm = 6.0
            if norm > 0:
                scale = target_norm / norm
                if scale > 1.5: scale = 1.5
                if scale < 0.8: scale = 0.8
                
                for idx in self.weights[o]:
                    self.weights[o][idx] *= scale
                    
        return f"[Sleep Phase] {pruned_total}個の不要なシナプス結合を枝刈りし、スケーリングを完了しました。（全結合数: {total_weights}）"