_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/nn/predictive.py",
    "//": "ファイルの日本語タイトル: 予測符号化スパイキング層",
    "//": "ファイルの目的や内容: 予測誤差のみを上位層へ伝達する生物学的メカニズム(Predictive Coding)。時系列の予測と重み更新のズレを修正し、正確なハビチュエーションを実現する。"
}

import random
from typing import List, Dict
from .module import SNNModule

class PredictiveSpikeLayer(SNNModule):
    def __init__(self, in_features: int, out_features: int, density: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.forward_weights: List[Dict[int, float]] = [{} for _ in range(in_features)]
        self.backward_weights: List[Dict[int, float]] = [{} for _ in range(out_features)]
        
        for i in range(in_features):
            num_conn = max(1, int(out_features * density))
            targets = random.sample(range(out_features), num_conn)
            for t in targets:
                self.forward_weights[i][t] = random.uniform(0.3, 0.8)
                
        for i in range(out_features):
            num_conn = max(1, int(in_features * density))
            targets = random.sample(range(in_features), num_conn)
            for t in targets:
                self.backward_weights[i][t] = random.uniform(0.1, 0.5)
                
        self.register_state("forward_weights")
        self.register_state("backward_weights")
        
        self.recent_out_spikes: List[int] = []

    def reset_state(self):
        super().reset_state()
        self.recent_out_spikes.clear()

    def forward(self, in_spikes: List[int], learning: bool = False, reward: float = 1.0) -> List[int]:
        # 1. トップダウン予測 (前回の出力状態から今回の入力を予測)
        pred_potentials = [0.0] * self.in_features
        for s in self.recent_out_spikes:
            if s < self.out_features:
                for t, w in self.backward_weights[s].items():
                    if t < self.in_features:
                        pred_potentials[t] += w
                    
        pred_threshold = 1.0
        predicted_in_spikes = set([i for i, p in enumerate(pred_potentials) if p > pred_threshold])
        
        # 2. 予測誤差の計算 (驚きのみが残る)
        error_spikes = [s for s in in_spikes if s not in predicted_in_spikes]
        
        # 3. ボトムアップ推論 (誤差のみを伝播)
        out_potentials = [0.0] * self.out_features
        for s in error_spikes:
            if s < self.in_features:
                for t, w in self.forward_weights[s].items():
                    if t < self.out_features:
                        out_potentials[t] += w
                    
        # 閾値を固定気味にして、エラーがなければ発火しないようにする
        dynamic_threshold = 0.8
        out_spikes = [i for i, p in enumerate(out_potentials) if p > dynamic_threshold]
        
        max_spikes = max(1, int(self.out_features * 0.25))
        if len(out_spikes) > max_spikes:
            out_spikes = sorted(out_spikes, key=lambda x: out_potentials[x], reverse=True)[:max_spikes]
            
        # 4. 学習
        if learning:
            out_set = set(out_spikes)
            in_set = set(in_spikes)
            
            # ボトムアップ: 誤差から正しい出力を導く
            for s in error_spikes:
                if s < self.in_features:
                    for t in list(self.forward_weights[s].keys()):
                        if t in out_set:
                            self.forward_weights[s][t] = min(3.0, self.forward_weights[s][t] + 0.15 * reward)
                        else:
                            self.forward_weights[s][t] = max(0.0, self.forward_weights[s][t] - 0.05)
                            if self.forward_weights[s][t] <= 0.0:
                                del self.forward_weights[s][t]
                                
            # トップダウン: 「前の出力」から「今回の実際の入力」を予測できるように強化
            for s in self.recent_out_spikes:
                if s < self.out_features:
                    # 予測を当てるため、必要なシナプスを新結合(構造的塑性)
                    for t in in_set:
                        if t not in self.backward_weights[s]:
                            if random.random() < 0.3:
                                self.backward_weights[s][t] = 0.5

                    for t in list(self.backward_weights[s].keys()):
                        if t in in_set:
                            # 予測が当たるようにLTPを強めに
                            self.backward_weights[s][t] = min(3.0, self.backward_weights[s][t] + 0.3 * reward)
                        else:
                            self.backward_weights[s][t] = max(0.0, self.backward_weights[s][t] - 0.1)
                            if self.backward_weights[s][t] <= 0.0:
                                del self.backward_weights[s][t]
                                
        self.recent_out_spikes = out_spikes
        return out_spikes