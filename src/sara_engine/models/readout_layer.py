_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/readout_layer.py",
    "//": "タイトル: Adam最適化機能付き読み出し層 (高精度・高速化チューニング版)",
    "//": "目的: 行列演算を排除し、スパースなリザーバー状態からの教師あり学習と、マージン強化・スケーリングの最適化により精度を向上させる。"
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
        
        # 直近の勾配への反応を良くするため beta1 を 0.90 に調整
        self.beta1 = 0.90
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        self.weights: List[Dict[int, float]] = [{} for _ in range(output_size)]
        self.m: List[Dict[int, float]] = [{} for _ in range(output_size)]
        self.v: List[Dict[int, float]] = [{} for _ in range(output_size)]
        
        # 初期化の幅を少し広げ、初期状態からスパースな結合を持たせることで探索を高速化
        limit = math.sqrt(3.0 / max(1, input_size))
        for o in range(output_size):
            for i in range(input_size):
                if random.random() < 0.2:
                    self.weights[o][i] = random.uniform(-limit, limit)
                    self.m[o][i] = 0.0
                    self.v[o][i] = 0.0

    def predict(self, active_hidden_indices: List[int]) -> List[float]:
        potentials = [0.0] * self.output_size
        if not active_hidden_indices:
            return potentials

        # スケーリングファクターの調整：活動ニューロン数に対するペナルティを緩和
        scale_factor = 20.0 / (len(active_hidden_indices) + 15.0)
        
        for o in range(self.output_size):
            s = 0.0
            w_o = self.weights[o]
            for idx in active_hidden_indices:
                if idx in w_o:
                    s += w_o[idx]
            potentials[o] = s * scale_factor
            
        max_p = max(potentials)
        if max_p > 0:
            mean_p = sum(potentials) / self.output_size
            for o in range(self.output_size):
                # 平均引き抑制を強め、コントラスト（S/N比）を高める
                potentials[o] -= 0.1 * mean_p
                if potentials[o] < -8.0: potentials[o] = -8.0
                if potentials[o] > 8.0: potentials[o] = 8.0
                
        return potentials

    def train_step(self, active_hidden_indices: List[int], target_label: int):
        if not active_hidden_indices:
            return
            
        self.t += 1
        # 学習率の減衰を緩やかにし、後半まで学習能力を維持
        current_lr = self.lr / (1.0 + 0.0005 * self.t)
        
        potentials = self.predict(active_hidden_indices)
        errors = [0.0] * self.output_size
        
        # 正解クラスは強い活動(3.0)を目指す
        if potentials[target_label] < 3.0:
            errors[target_label] = 3.0 - potentials[target_label]
            
        # ソフト・ネガティブ: 不正解クラスはより強く(-1.0)押し戻す
        for o in range(self.output_size):
            if o != target_label and potentials[o] > -0.5:
                errors[o] = -1.0 - potentials[o]
                
        for o in range(self.output_size):
            err = errors[o]
            # エラーの不感帯を広げて無駄なスパース辞書の更新(計算)を省き高速化
            if abs(err) <= 0.05:
                continue
                
            w_o = self.weights[o]
            m_o = self.m[o]
            v_o = self.v[o]
            
            for idx in active_hidden_indices:
                if idx not in w_o:
                    w_o[idx] = 0.0
                    m_o[idx] = 0.0
                    v_o[idx] = 0.0
                
                g = err 
                
                m_o[idx] = self.beta1 * m_o[idx] + (1 - self.beta1) * g
                v_o[idx] = self.beta2 * v_o[idx] + (1 - self.beta2) * (g * g)
                
                m_hat = m_o[idx] / (1 - (self.beta1 ** min(self.t, 1000)))
                v_hat = v_o[idx] / (1 - (self.beta2 ** min(self.t, 1000)))
                
                w_o[idx] += current_lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
                
                if w_o[idx] < -6.0: w_o[idx] = -6.0
                if w_o[idx] > 6.0: w_o[idx] = 6.0

    def sleep_phase(self, prune_rate: float = 0.05) -> str:
        pruned_total = 0
        total_weights = 0
        
        for o in range(self.output_size):
            w_o = self.weights[o]
            # 自然減衰を少し強め、不要な結合の淘汰を促進
            for idx in list(w_o.keys()):
                w_o[idx] *= 0.995
                
            active_weights = [abs(w) for w in w_o.values() if abs(w) > 1e-6]
            if not active_weights:
                continue
                
            active_weights.sort()
            prune_idx = int(len(active_weights) * prune_rate)
            threshold = active_weights[prune_idx] if prune_idx < len(active_weights) else 0.0
            
            keys_to_remove = []
            sq_sum = 0.0
            
            for idx, w in w_o.items():
                total_weights += 1
                if abs(w) < threshold:
                    keys_to_remove.append(idx)
                    pruned_total += 1
                else:
                    sq_sum += w * w
                    
            for idx in keys_to_remove:
                del w_o[idx]
                if idx in self.m[o]: del self.m[o][idx]
                if idx in self.v[o]: del self.v[o][idx]
                
            norm = math.sqrt(sq_sum)
            # ターゲットノルムを上げて表現力を維持
            target_norm = 8.0
            if norm > 0:
                scale = target_norm / norm
                if scale > 2.0: scale = 2.0
                if scale < 0.5: scale = 0.5
                
                for idx in w_o:
                    w_o[idx] *= scale
                    
        return f"[Sleep Phase] {pruned_total}個の不要なシナプス結合を枝刈りし、スケーリングを完了しました。（全結合数: {total_weights}）"