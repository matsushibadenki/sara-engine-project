_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/nn/rstdp.py",
    "//": "ファイルの日本語タイトル: 報酬変調STDP層 (R-STDP)",
    "//": "ファイルの目的や内容: 誤差逆伝播法(BP)に依存せず、適格度トレースと遅延報酬を用いた「3要素学習則」によって大域的最適化を行う強化学習用SNNモジュール。"
}

import random
from typing import List, Dict, Tuple
from .module import SNNModule

class RewardModulatedLinearSpike(SNNModule):
    """
    R-STDP (Reward-Modulated STDP) に基づく線形スパイク層。
    3要素学習則を採用し、
    1. Pre-synaptic (シナプス前発火)
    2. Post-synaptic (シナプス後発火)
    3. Neuromodulator (ドーパミンなどの遅延報酬シグナル)
    の組み合わせにより、BPなしで強化学習を実現する。
    """
    def __init__(self, in_features: int, out_features: int, density: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights: List[Dict[int, float]] = [{} for _ in range(in_features)]
        for i in range(in_features):
            num_connections = max(1, int(out_features * density))
            targets = random.sample(range(out_features), num_connections)
            for t in targets:
                self.weights[i][t] = random.uniform(0.1, 1.0)
                
        self.register_state("weights")
        
        # 適格度トレース (Eligibility Trace): 直近の発火ペアの「痕跡」を保持
        # キー: (pre_id, post_id), 値: trace_value
        self.eligibility_traces: Dict[Tuple[int, int], float] = {}
        self.trace_decay = 0.9  # 時間経過によるトレースの自然減衰率

    def forward(self, spikes: List[int], learning: bool = False) -> List[int]:
        # 学習時、過去のトレースを自然減衰させる
        if learning:
            keys_to_remove = []
            for k in self.eligibility_traces:
                self.eligibility_traces[k] *= self.trace_decay
                if self.eligibility_traces[k] < 0.01:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del self.eligibility_traces[k]

        # 膜電位の計算と発火判定
        potentials = [0.0] * self.out_features
        for s in spikes:
            if s < self.in_features:
                for target, weight in self.weights[s].items():
                    if target < self.out_features:
                        potentials[target] += weight
                        
        active_spikes = [(i, p) for i, p in enumerate(potentials) if p > 0.5]
        active_spikes.sort(key=lambda x: x[1], reverse=True)
        max_spikes = max(1, int(self.out_features * 0.2)) # Top-K発火
        out_spikes = [i for i, p in active_spikes[:max_spikes]]
        
        # 適格度トレースの記録（この時点ではまだ重みは更新しない）
        # シナプス前とシナプス後が共に発火した場合、結合の痕跡を残す
        if learning:
            for pre in spikes:
                if pre < self.in_features:
                    for post in out_spikes:
                        current_trace = self.eligibility_traces.get((pre, post), 0.0)
                        self.eligibility_traces[(pre, post)] = current_trace + 1.0
                        
        return out_spikes

    def apply_reward(self, reward: float, learning_rate: float = 0.1) -> None:
        """
        環境から遅延報酬（スカラー値）を受け取り、シナプス荷重を更新する。
        """
        keys_to_remove = []
        for (pre, post), trace in self.eligibility_traces.items():
            if post in self.weights[pre]:
                # 3要素学習則: Δw = 学習率 × トレース量 × 報酬シグナル
                delta_w = learning_rate * trace * reward
                new_w = self.weights[pre][post] + delta_w
                # 発火率の暴走を防ぐため荷重をクリッピング
                self.weights[pre][post] = max(0.0, min(3.0, new_w))
                
            # 報酬適用後はトレースを消費したとみなして削除
            keys_to_remove.append((pre, post))
            
        for k in keys_to_remove:
            del self.eligibility_traces[k]

    def reset_state(self):
        super().reset_state()
        self.eligibility_traces.clear()