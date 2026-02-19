_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/readout_layer.py",
    "//": "タイトル: Adam最適化機能付き読み出し層 (時空間・永続化対応版)",
    "//": "目的: 時間と空間の特徴を分離して受け取り、学習済みモデルの重みと状態をJSONで高速に保存・復元できるようにする。"
}

import math
import random
import json
from typing import List, Dict

class ReadoutLayer:
    def __init__(self, input_size: int, output_size: int, learning_rate: float = 0.002):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = learning_rate
        self.t = 0
        
        self.beta1 = 0.90
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        self.weights: List[Dict[int, float]] = [{} for _ in range(output_size)]
        self.m: List[Dict[int, float]] = [{} for _ in range(output_size)]
        self.v: List[Dict[int, float]] = [{} for _ in range(output_size)]
        
        limit = math.sqrt(3.0 / max(1, input_size))
        for o in range(output_size):
            n_init = int(input_size * 0.15)
            for i in random.sample(range(input_size), n_init):
                self.weights[o][i] = random.uniform(-limit, limit)
                self.m[o][i] = 0.0
                self.v[o][i] = 0.0

    def predict(self, active_hidden_indices: List[int]) -> List[float]:
        potentials = [0.0] * self.output_size
        if not active_hidden_indices:
            return potentials

        scale_factor = 10.0 / (math.sqrt(len(active_hidden_indices)) + 1.0)
        
        for o in range(self.output_size):
            s = 0.0
            w_o = self.weights[o]
            for idx in active_hidden_indices:
                if idx in w_o:
                    s += w_o[idx]
            potentials[o] = s * scale_factor
            
        max_p = max(potentials)
        if max_p > -999.0:
            mean_p = sum(potentials) / self.output_size
            for o in range(self.output_size):
                potentials[o] -= 0.15 * mean_p
                if potentials[o] < -10.0: potentials[o] = -10.0
                if potentials[o] > 10.0: potentials[o] = 10.0
                
        return potentials

    def train_step(self, active_hidden_indices: List[int], target_label: int):
        if not active_hidden_indices:
            return
            
        self.t += 1
        current_lr = self.lr / (1.0 + 0.0002 * self.t)
        
        potentials = self.predict(active_hidden_indices)
        errors = [0.0] * self.output_size
        
        if potentials[target_label] < 4.0:
            errors[target_label] = 4.0 - potentials[target_label]
            
        for o in range(self.output_size):
            if o != target_label and potentials[o] > -1.5:
                errors[o] = -1.5 - potentials[o]
                
        for o in range(self.output_size):
            err = errors[o]
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
            target_norm = 10.0
            if norm > 0:
                scale = target_norm / norm
                if scale > 2.0: scale = 2.0
                if scale < 0.5: scale = 0.5
                
                for idx in w_o:
                    w_o[idx] *= scale
                    
        return f"[Sleep Phase] {pruned_total}個の不要なシナプス結合を枝刈りし、スケーリングを完了しました。（全結合数: {total_weights}）"

    def save_model(self, filepath: str):
        """モデルの重みと状態をJSONファイルに保存する"""
        data = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "lr": self.lr,
            "t": self.t,
            "weights": [{str(k): v for k, v in w.items()} for w in self.weights],
            "m": [{str(k): v for k, v in m_dict.items()} for m_dict in self.m],
            "v": [{str(k): v for k, v in v_dict.items()} for v_dict in self.v]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"[Model] {filepath} にモデルを保存しました。")

    def load_model(self, filepath: str):
        """JSONファイルからモデルの重みと状態を読み込む"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.input_size = data["input_size"]
        self.output_size = data["output_size"]
        self.lr = data["lr"]
        self.t = data["t"]
        self.weights = [{int(k): v for k, v in w.items()} for w in data["weights"]]
        self.m = [{int(k): v for k, v in m_dict.items()} for m_dict in data["m"]]
        self.v = [{int(k): v for k, v in v_dict.items()} for v_dict in data["v"]]
        print(f"[Model] {filepath} からモデルを読み込みました。")