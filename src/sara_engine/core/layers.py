_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/layers.py",
    "//": "タイトル: Dynamic Liquid Layer (Homeostasis & R-STDP Core)",
    "//": "目的: Rust実装およびPythonフォールバックの両方で、報酬変調型STDP（R-STDP）をサポートするよう拡張。"
}

import random
from typing import List, Optional, Tuple, Dict

# Rust拡張のインポート試行
try:
    from .. import sara_rust_core  # type: ignore
    RUST_AVAILABLE = True
except ImportError:
    try:
        import sara_rust_core  # type: ignore
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False


class DynamicLiquidLayer:
    def __init__(self, input_size: int, hidden_size: int, decay: float, 
                 density: float = 0.05, input_scale: float = 1.0, 
                 rec_scale: float = 0.8, feedback_scale: float = 0.5,
                 use_rust: Optional[bool] = None,
                 target_rate: float = 0.05):
        
        self.size = hidden_size
        self.input_size = input_size
        self.decay = decay
        self.density = density
        self.input_scale = input_scale
        self.rec_scale = rec_scale
        self.feedback_scale = feedback_scale
        self.target_rate = target_rate
        
        # Rust使用の自動判定
        if use_rust is None:
            self.use_rust = RUST_AVAILABLE
        else:
            self.use_rust = use_rust and RUST_AVAILABLE

        if self.use_rust:
            # Rust実装の初期化（すべての重みと状態はRust側で管理される）
            self.core = sara_rust_core.RustLiquidLayer(
                input_size, hidden_size, decay, density, feedback_scale
            )
            print("DynamicLiquidLayer: Rust core initialized.")
        else:
            # Python実装（辞書型疎結合）
            print("DynamicLiquidLayer: Python fallback mode.")
            self.v = [0.0] * hidden_size
            self.refractory = [0.0] * hidden_size
            self.dynamic_thresh = [1.0] * hidden_size
            self.trace = [0.0] * hidden_size  # STDP用トレース
            
            # 入力重み: 辞書のリスト [input_idx] -> {hidden_idx: weight, ...}
            self.in_weights: List[Dict[int, float]] = [{} for _ in range(input_size)]
            for i in range(input_size):
                # 密度に基づいて接続先をランダム選択
                n_connect = int(hidden_size * density)
                if n_connect > 0:
                    targets = random.sample(range(hidden_size), n_connect)
                    for t in targets:
                        self.in_weights[i][t] = random.uniform(-input_scale, input_scale)
            
            # 再帰重み: 辞書のリスト [hidden_idx] -> {hidden_idx: weight, ...}
            self.rec_weights: List[Dict[int, float]] = [{} for _ in range(hidden_size)]
            rec_density = 0.1
            for i in range(hidden_size):
                n_connect = int(hidden_size * rec_density)
                if n_connect > 0:
                    # 自分自身への結合は避ける
                    candidates = [x for x in range(hidden_size) if x != i]
                    if len(candidates) >= n_connect:
                        targets = random.sample(candidates, n_connect)
                        for t in targets:
                            self.rec_weights[i][t] = random.uniform(-rec_scale, rec_scale)

            # フィードバック結合（ランダム）
            self.feedback_map: List[List[int]] = []
            for _ in range(hidden_size): # フィードバック入力サイズと仮定
                n = int(hidden_size * 0.05)
                self.feedback_map.append(random.sample(range(hidden_size), n))

    def get_state(self) -> Tuple[List[float], List[float]]:
        """現在の膜電位と動的閾値の状態を取得する"""
        if self.use_rust:
            if hasattr(self.core, 'get_state'):
                return self.core.get_state()
            return [], []
        return self.v, self.dynamic_thresh

    def forward_with_feedback(self, active_inputs: List[int], prev_active_hidden: List[int], 
                              feedback_active: List[int] = [], attention_signal: List[int] = [],
                              learning: bool = False, reward: float = 1.0) -> List[int]:
        """フィードバック付きでforwardを実行するエイリアス"""
        return self.forward(
            active_inputs=active_inputs, 
            prev_active_hidden=prev_active_hidden, 
            feedback_active=feedback_active,
            attention_signal=attention_signal,
            learning=learning,
            reward=reward
        )

    def forward(self, active_inputs: List[int], prev_active_hidden: List[int], 
                feedback_active: List[int] = [], attention_signal: List[int] = [],
                learning: bool = False, reward: float = 1.0) -> List[int]:
        
        if self.use_rust:
            # Rust側へは単純な整数のリストとrewardを渡す
            return self.core.forward(
                active_inputs, prev_active_hidden, feedback_active, attention_signal, learning, float(reward)
            )
        
        # --- Python純粋実装 ---

        # 1. 減衰と不応期の更新
        for i in range(self.size):
            self.v[i] *= self.decay
            if self.refractory[i] > 0.0:
                self.refractory[i] -= 1.0
            self.trace[i] *= 0.95 # トレースの減衰

        # 2. 入力スパイクの統合 (疎結合なのでO(k)で高速)
        for inp_idx in active_inputs:
            if inp_idx < len(self.in_weights):
                for target, weight in self.in_weights[inp_idx].items():
                    self.v[target] += weight

        # 3. 再帰スパイクの統合
        for hid_idx in prev_active_hidden:
            if hid_idx < len(self.rec_weights):
                for target, weight in self.rec_weights[hid_idx].items():
                    self.v[target] += weight
        
        # 4. フィードバックとAttention信号
        for fb_idx in feedback_active:
            if fb_idx < len(self.feedback_map):
                for target in self.feedback_map[fb_idx]:
                    self.v[target] += self.feedback_scale
        
        for att_idx in attention_signal:
            if att_idx < self.size:
                self.v[att_idx] += 1.5 # 強い興奮

        # 5. 発火判定 (Winner-Take-All的な抑制と動的閾値)
        fired_indices = []
        candidates = []
        
        for i in range(self.size):
            if self.v[i] >= self.dynamic_thresh[i] and self.refractory[i] <= 0:
                candidates.append(i)
        
        # 発火数制限 (Homeostasisの一環)
        max_spikes = int(self.size * 0.15)
        if len(candidates) > max_spikes:
            # 電位が高い順にソートして上位のみ発火
            candidates.sort(key=lambda x: self.v[x], reverse=True)
            fired_indices = candidates[:max_spikes]
        else:
            fired_indices = candidates
            
        fired_set = set(fired_indices)

        # 6. 発火後の処理 (リセット、不応期、閾値調整)
        for i in range(self.size):
            if i in fired_set:
                self.v[i] = 0.0
                self.refractory[i] = random.uniform(2.0, 5.0)
                self.trace[i] += 1.0
                # 発火したので閾値を上げる
                self.dynamic_thresh[i] += 0.05
            else:
                # 発火しなかったので閾値を下げる（忘却）
                self.dynamic_thresh[i] += (self.target_rate - 0.05) * 0.01
            
            # 閾値クリッピング
            if self.dynamic_thresh[i] < 0.5: self.dynamic_thresh[i] = 0.5
            if self.dynamic_thresh[i] > 5.0: self.dynamic_thresh[i] = 5.0

        # 7. R-STDP (報酬変調型 STDP) - 学習
        if learning and prev_active_hidden:
            for pre_id in prev_active_hidden:
                if pre_id < len(self.rec_weights):
                    # 既存の接続のみ更新
                    for target_id in self.rec_weights[pre_id].keys():
                        if target_id in fired_set:
                            # LTP (Long-Term Potentiation): 前→後 の順で発火 (報酬変調)
                            self.rec_weights[pre_id][target_id] += 0.02 * reward
                            if self.rec_weights[pre_id][target_id] > 2.0:
                                self.rec_weights[pre_id][target_id] = 2.0
                        else:
                            # LTD (Long-Term Depression): 前のみ発火 (報酬変調)
                            self.rec_weights[pre_id][target_id] -= 0.005 * reward
                            if self.rec_weights[pre_id][target_id] < -2.0:
                                self.rec_weights[pre_id][target_id] = -2.0

        return fired_indices

    def reset(self):
        if self.use_rust:
            self.core.reset()
        else:
            self.v = [0.0] * self.size
            self.refractory = [0.0] * self.size
            self.dynamic_thresh = [1.0] * self.size
            self.trace = [0.0] * self.size