_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/nn/predictive.py",
    "//": "ファイルの日本語タイトル: 予測符号化スパイキング層",
    "//": "ファイルの目的や内容: 予測誤差のみを上位層へ伝達する生物学的メカニズム(Predictive Coding)。推論時にも時系列履歴を正しく更新するよう修正。"
}

import random
from typing import List, Dict, Tuple
from .module import SNNModule

try:
    from sara_engine.sara_rust_core import CausalSynapses
except ImportError:
    CausalSynapses = None

# =====================================================================
# [1] 既存の双方向・空間的予測レイヤー (Legacy / Spatial)
# =====================================================================

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


# =====================================================================
# [2] フェーズ2 時系列・誤差主導予測レイヤー (Phase 2 / Temporal)
# =====================================================================

class PredictiveCodingLayer(SNNModule):
    """
    Phase 2: Predictive Coding (Temporal)
    入力されたスパイクに対し、過去の履歴から「予測」を行い、
    予測できなかった「誤差（サプライズ）」のスパイクのみを抽出して学習・上位伝播させます。
    RustのCausalSynapsesを利用して高速に動作します。
    """
    def __init__(self, max_delay: int = 10, learning_rate: float = 0.05, threshold: float = 0.5):
        super().__init__()
        self.max_delay = max_delay
        self.learning_rate = learning_rate
        self.threshold = threshold
        
        if CausalSynapses is None:
            raise RuntimeError("sara_rust_core is not available. Please run `pip install -e .`")
            
        self.synapses = CausalSynapses(max_delay=max_delay)
        
        # ネットワークの時系列状態
        self.spike_history: List[List[int]] = []
        self.register_state("spike_history")

    def forward(self, actual_spikes: List[int], learning: bool = True) -> Tuple[List[int], float]:
        """
        予測と抑制を行い、誤差スパイクのみを返す。
        戻り値: (上位へ伝播すべき誤差スパイクのリスト, 今回の予測誤差率)
        """
        if not self.spike_history:
            # 推論時(learning=False)でも最初の履歴を確実に保存するよう修正
            self.spike_history.insert(0, actual_spikes)
            return actual_spikes, 1.0

        if learning:
            # Rustコアで予測・抑制・誤差主導STDPを一括処理
            error_spikes, error_rate = self.synapses.predict_and_learn(
                self.spike_history, 
                actual_spikes, 
                self.learning_rate, 
                self.threshold
            )
            # 時系列履歴の更新
            self.spike_history.insert(0, actual_spikes)
            if len(self.spike_history) > self.max_delay:
                self.spike_history.pop()
                
            return error_spikes, error_rate
        else:
            # 推論時: 予測の生成と抑制（フィルタリング）のみ実行
            potentials = self.synapses.calculate_potentials(self.spike_history)
            predicted = [t_id for t_id, pot in potentials.items() if pot >= self.threshold]
            
            error_spikes = list(set(actual_spikes) - set(predicted))
            error_rate = len(error_spikes) / max(1, len(actual_spikes))
            
            self.spike_history.insert(0, actual_spikes)
            if len(self.spike_history) > self.max_delay:
                self.spike_history.pop()
                
            return error_spikes, error_rate

    def predict_next(self) -> List[int]:
        """現在の履歴から、次のタイムステップで発火するスパイクを生成・予測する"""
        potentials = self.synapses.calculate_potentials(self.spike_history)
        predicted = [t_id for t_id, pot in potentials.items() if pot >= self.threshold]
        return predicted
        
    def reset_state(self) -> None:
        """時系列履歴の初期化"""
        self.spike_history.clear()
        super().reset_state()