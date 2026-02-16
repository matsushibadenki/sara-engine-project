# パス: src/sara_engine/learning/stdp.py
# タイトル: STDP（スパイクタイミング依存可塑性）学習レイヤー
# 目的: 厳格なシナプス・スケーリングと競合学習を組み合わせ、入力パターンを明確に分化・自己組織化させる。さらに報酬変調(R-STDP)と構造的可塑性(スパース化)により精度と省エネを極限まで高める。
# {
#     "//": "行列演算・誤差逆伝播を完全排除。イベント駆動とスパース接続による超省エネ設計。"
# }
import random

class STDPLayer:
    def __init__(self, num_inputs: int, num_outputs: int, threshold: float = 2.0):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        # 構造的可塑性: 疎結合(Sparse)な辞書表現でシナプスを管理し、ゼロ重みの演算を完全にスキップ（超省エネ）
        # outputs[j] が受け取る inputs[i] の重み
        self.synapses = []
        for _ in range(num_outputs):
            # 初期状態ではランダムに30%の結線のみを持つ（省エネ）
            connections = {}
            for i in range(num_inputs):
                if random.random() < 0.3:
                    connections[i] = random.uniform(0.1, 0.5)
            self.synapses.append(connections)
        
        self.A_plus = 0.15   # 強化（LTP）の学習率
        self.A_minus = 0.05  # 抑圧（LTD）の学習率
        
        self.potentials = [0.0] * num_outputs
        self.leak_rate = 0.9

        # Intrinsic Plasticity（適応型閾値）のパラメータ
        self.base_threshold = threshold
        self.thresholds = [threshold] * num_outputs
        self.theta_plus = 0.5   # 発火時の閾値上昇（疲労）
        self.theta_decay = 0.05 # 基準値へ戻る自然減衰
        
        self.target_weight_sum = 2.5
        self.prune_threshold = 0.01  # この値以下のシナプスは刈り取る（構造的可塑性）

    def process_step(self, input_spikes: list[int], reward: float = 1.0) -> tuple[list[int], list[float]]:
        # reward: 1.0 (正常LTP), マイナス値 (LTD反転: 罰) -> R-STDPによる精度向上
        output_spikes = [0] * self.num_outputs
        
        # 入力スパイクをイベント（発火したインデックス）に変換
        active_inputs = [i for i, s in enumerate(input_spikes) if s == 1]
        
        # 1. 膜電位の計算と発火判定 (イベント駆動: 発火した入力のみループ)
        max_pot = -1.0
        winner_idx = -1
        
        for j in range(self.num_outputs):
            self.potentials[j] *= self.leak_rate
            
            for i in active_inputs:
                if i in self.synapses[j]:
                    self.potentials[j] += self.synapses[j][i]
                    
            if self.potentials[j] >= self.thresholds[j] and self.potentials[j] > max_pot:
                max_pot = self.potentials[j]
                winner_idx = j
                
        # 2. 強力なWinner-Take-All
        if winner_idx != -1:
            output_spikes[winner_idx] = 1
            self.potentials[winner_idx] = 0.0
            
            # Intrinsic Plasticity: 勝者の閾値を上げて連続的な独占を防ぐ
            self.thresholds[winner_idx] += self.theta_plus
            
            # 敗者の側抑制
            for j in range(self.num_outputs):
                if j != winner_idx:
                    self.potentials[j] = -1.0

        # 3. 適応型閾値の減衰
        for j in range(self.num_outputs):
            self.thresholds[j] += (self.base_threshold - self.thresholds[j]) * self.theta_decay

        # 4. R-STDP (報酬変調型STDP) と構造的可塑性
        if winner_idx != -1:
            j = winner_idx
            current_synapses = self.synapses[j]
            
            active_set = set(active_inputs)
            keys_to_remove = []
            
            for i in list(current_synapses.keys()):
                if i in active_set:
                    # 報酬がプラスなら強化、マイナスなら減弱 (R-STDP)
                    current_synapses[i] += self.A_plus * reward
                else:
                    current_synapses[i] -= self.A_minus
                
                # 構造的可塑性: 弱すぎるシナプスを物理的に刈り取る（省エネ）
                if current_synapses[i] < self.prune_threshold:
                    keys_to_remove.append(i)
            
            for i in keys_to_remove:
                del current_synapses[i]
                
            # ランダムなシナプス新生（探索と自己組織化の維持）
            if reward > 0 and random.random() < 0.1 and active_inputs:
                new_i = random.choice(active_inputs)
                if new_i not in current_synapses:
                    current_synapses[new_i] = random.uniform(0.1, 0.2)
                    
            # 厳格なSynaptic Scaling（重みの総量規制）
            current_sum = sum(current_synapses.values())
            if current_sum > 0:
                scale_factor = self.target_weight_sum / current_sum
                for i in current_synapses:
                    current_synapses[i] *= scale_factor
                    
                    # 最終的な結合荷重のクリッピング
                    if current_synapses[i] > 1.0:
                        current_synapses[i] = 1.0

        return output_spikes, list(self.potentials)