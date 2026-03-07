# ファイルメタ情報
# ディレクトリパス: src/sara_engine/models/readout_layer.py
# ファイルの日本語タイトル: スパイク読み出し層 (Top-1 Hinge-Loss STDP / 生物学的WTA)
# ファイルの目的や内容: Winner-Take-All（側抑制）の原理を応用し、最大の競合クラス（Top-1）に対してのみマージン更新を行うPassive-Aggressiveアルゴリズムを実装。独立サンプルの分類タスクに悪影響を与えていた不応期（Refractory）の無効化オプションを追加し、精度95%以上の壁を突破する。
import random
import math
from typing import List, Dict, Optional
from ..learning.homeostasis import AdaptiveThresholdHomeostasis

class SpikeReadoutLayer:
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        learning_rate: float = 0.01,
        use_refractory: bool = False,
        use_homeostasis: bool = False,
        homeostasis_target_rate: Optional[float] = None,
        homeostasis_adaptation_rate: float = 0.02,
        homeostasis_decay: float = 0.995,
        homeostasis_strength: float = 8.0,
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        
        # LLM等の時系列生成タスクではTrue、画像分類等の独立タスクではFalseを使用する
        self.use_refractory = use_refractory
        
        self.W: List[Dict[int, float]] = [{} for _ in range(d_model)]
        self.b: List[float] = [0.0] * vocab_size
        
        self.refractory_states: Dict[int, int] = {}
        self.refractory_duration = 2
        self.use_homeostasis = use_homeostasis
        self.homeostasis_strength = homeostasis_strength
        self.homeostasis: Optional[AdaptiveThresholdHomeostasis] = None
        if use_homeostasis:
            target_rate = homeostasis_target_rate
            if target_rate is None:
                target_rate = 1.0 / max(1, vocab_size)
            self.homeostasis = AdaptiveThresholdHomeostasis(
                target_rate=target_rate,
                adaptation_rate=homeostasis_adaptation_rate,
                decay=homeostasis_decay,
                min_threshold=0.0,
                max_threshold=1.5,
                global_weight=0.35,
            )
        
        # 表現力を制限しないよう上下限を広げる
        self.w_max = 50.0
        self.w_min = -50.0 

    def active_synapse_count(self) -> int:
        total = 0
        for row in self.W:
            total += len(row)
        return total

    def prune_weights(self, min_abs_weight: float = 0.01) -> int:
        pruned = 0
        for row in self.W:
            to_delete = [token_id for token_id, weight in row.items() if abs(weight) < min_abs_weight]
            for token_id in to_delete:
                del row[token_id]
                pruned += 1
        return pruned

    def forward(self, spikes: List[int], target_token: Optional[int] = None, learning: bool = True) -> int:
        if self.use_refractory:
            for tid in list(self.refractory_states.keys()):
                self.refractory_states[tid] -= 1
                if self.refractory_states[tid] <= 0:
                    del self.refractory_states[tid]

        # 膜電位の計算 (バイアス項 + スパイク加算)
        potentials = list(self.b)
        for s in spikes:
            if s < len(self.W):
                for token_id, weight in self.W[s].items():
                    potentials[token_id] += weight

        adjusted_potentials = list(potentials)
        if self.homeostasis is not None:
            for token_id in range(self.vocab_size):
                adjusted_potentials[token_id] -= (
                    self.homeostasis.get_threshold(token_id, 0.0) * self.homeostasis_strength
                )
                
        if not learning and self.use_refractory:
            for tid in self.refractory_states:
                if tid < self.vocab_size:
                    adjusted_potentials[tid] = -9999.0 
                
        predicted_token = 0
        max_potential = -9999.0
        
        for i in range(self.vocab_size):
            if adjusted_potentials[i] > max_potential:
                max_potential = adjusted_potentials[i]
                predicted_token = i
        
        if max_potential <= -9000.0 and len(spikes) == 0:
            predicted_token = random.randint(0, self.vocab_size - 1)
            
        if not learning and self.use_refractory:
            self.refractory_states[predicted_token] = self.refractory_duration

        if self.homeostasis is not None:
            active_ids = [predicted_token] if len(spikes) > 0 else []
            self.homeostasis.update(active_ids, population_size=self.vocab_size)

        # Top-1 Hinge-Loss STDP (Crammer & Singer Multiclass PA)
        # 生物学的な側抑制（Winner-Take-All）に相当し、発散を完全に防ぐ
        if learning and target_token is not None and len(spikes) > 0:
            norm_sq = float(len(spikes))
            
            # 最大のライバル（不正解クラスの中で最も電位が高いもの）を探索
            max_wrong_token = -1
            max_wrong_potential = -999999.0
            
            for i in range(self.vocab_size):
                if i != target_token:
                    if adjusted_potentials[i] > max_wrong_potential:
                        max_wrong_potential = adjusted_potentials[i]
                        max_wrong_token = i
            
            # 学習率のスケール係数と要求マージンの動的調整
            C = self.learning_rate * 50.0
            
            # スパイク数に応じて要求マージンをスケーリングし、多次元空間での埋没を防ぐ
            margin_scale = math.sqrt(norm_sq) / 10.0 if norm_sq > 100 else 1.0
            target_margin = 20.0 * margin_scale
            
            # Top-1 クラスに対するヒンジロス
            if max_wrong_token != -1:
                loss = target_margin - (adjusted_potentials[target_token] - max_wrong_potential)
                
                if loss > 0:
                    # 正解クラスと不正解クラスの2つを更新するため、分母は 2 * norm_sq
                    tau = min(C, loss / (2.0 * norm_sq + 1.0))
                    
                    # バイアスの更新
                    self.b[target_token] += tau
                    self.b[max_wrong_token] -= tau
                    
                    # 恒常性（Weight Decay / L2正則化）係数
                    decay = 1e-5
                    
                    # シナプス結合の更新
                    for s in spikes:
                        if s < len(self.W):
                            # [LTP] 正解ルートの増強
                            w_target = self.W[s].get(target_token, 0.0)
                            new_w_target = w_target * (1.0 - decay) + tau
                            if new_w_target > self.w_max:
                                new_w_target = self.w_max
                            
                            if -0.001 < new_w_target < 0.001:
                                if target_token in self.W[s]:
                                    del self.W[s][target_token]
                            else:
                                self.W[s][target_token] = new_w_target
                            
                            # [LTD] 最大ライバルルートの抑圧
                            w_wrong = self.W[s].get(max_wrong_token, 0.0)
                            new_w_wrong = w_wrong * (1.0 - decay) - tau
                            if new_w_wrong < self.w_min:
                                new_w_wrong = self.w_min
                                
                            if -0.001 < new_w_wrong < 0.001:
                                if max_wrong_token in self.W[s]:
                                    del self.W[s][max_wrong_token]
                            else:
                                self.W[s][max_wrong_token] = new_w_wrong

        return predicted_token
