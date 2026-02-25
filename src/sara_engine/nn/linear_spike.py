_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/nn/linear_spike.py",
    "//": "ファイルの日本語タイトル: スパイキング全結合層",
    "//": "ファイルの目的や内容: SNNModuleを継承した、STDP学習付きの疎結合スパイキング線形層。スパイク消失と全発火のバランスを取るよう重みと閾値を調整。"
}

import random
from typing import List, Dict
from .module import SNNModule

class LinearSpike(SNNModule):
    def __init__(self, in_features: int, out_features: int, density: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # weights[pre][post] = weight
        self.weights: List[Dict[int, float]] = [{} for _ in range(in_features)]
        for i in range(in_features):
            num_conn = max(1, int(out_features * density))
            targets = random.sample(range(out_features), num_conn)
            for t in targets:
                # 初期スパイクが消失しないよう、適度な重みを持たせる
                self.weights[i][t] = random.uniform(0.3, 0.8)
                
        self.register_state("weights")

    def reset_state(self):
        pass

    def forward(self, in_spikes: List[int], learning: bool = False) -> List[int]:
        potentials = [0.0] * self.out_features
        
        # Integrate (スパイクの統合)
        for s in in_spikes:
            if s < self.in_features:
                for t, w in self.weights[s].items():
                    potentials[t] += w
                    
        # Fire (閾値判定)
        # 閾値を少し下げて、信号が奥の層まで伝播するようにする
        dynamic_threshold = max(0.5, len(in_spikes) * 0.1)
        out_spikes = [i for i, p in enumerate(potentials) if p > dynamic_threshold]
        
        # 出力が多すぎる場合は上位K個（Winner-Takes-All的）に絞り全発火を防ぐ
        max_spikes = max(1, int(self.out_features * 0.25))
        if len(out_spikes) > max_spikes:
            out_spikes = sorted(out_spikes, key=lambda x: potentials[x], reverse=True)[:max_spikes]
        
        # STDP Learning (局所学習)
        if learning:
            out_set = set(out_spikes)
            for s in in_spikes:
                if s < self.in_features:
                    for t in list(self.weights[s].keys()):
                        if t in out_set:
                            # LTP (長期増強)
                            self.weights[s][t] = min(3.0, self.weights[s][t] + 0.1)
                        else:
                            # LTD (長期抑圧)
                            self.weights[s][t] = max(0.0, self.weights[s][t] - 0.05)
                            if self.weights[s][t] <= 0.0:
                                del self.weights[s][t]
        
        return out_spikes