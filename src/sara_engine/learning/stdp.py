# パス: src/sara_engine/learning/stdp.py
# タイトル: STDP（スパイクタイミング依存可塑性）学習レイヤー
# 目的: 厳格なシナプス・スケーリングと競合学習を組み合わせ、入力パターンを明確に分化・自己組織化させる。
# {
#     "//": "行列演算・誤差逆伝播を完全排除。勝者ニューロンに対する即時的な重み正規化を強制。"
# }
import random

class STDPLayer:
    def __init__(self, num_inputs: int, num_outputs: int, threshold: float = 2.0):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        # 行列演算を使わず、2次元リストでシナプス結合荷重を管理
        self.weights = [[random.uniform(0.1, 0.5) for _ in range(num_outputs)] for _ in range(num_inputs)]
        
        self.A_plus = 0.15   # 強化（LTP）の学習率
        self.A_minus = 0.05  # 抑圧（LTD）の学習率
        
        self.potentials = [0.0] * num_outputs
        self.leak_rate = 0.9

        # Intrinsic Plasticity（適応型閾値）のパラメータ
        self.base_threshold = threshold
        self.thresholds = [threshold] * num_outputs
        self.theta_plus = 0.5   # 発火時の閾値上昇（疲労）
        self.theta_decay = 0.05 # 基準値へ戻る自然減衰
        
        # 厳格なシナプス・スケーリングの目標値
        # 1つのニューロンが持てる重みの総和を制限し、一部の入力のみに特化させる
        self.target_weight_sum = 2.5

    def process_step(self, input_spikes: list[int]) -> tuple[list[int], list[float]]:
        output_spikes = [0] * self.num_outputs
        
        # 1. 膜電位の計算と発火判定
        max_pot = -1.0
        winner_idx = -1
        
        for j in range(self.num_outputs):
            self.potentials[j] *= self.leak_rate
            for i in range(self.num_inputs):
                if input_spikes[i] == 1:
                    self.potentials[j] += self.weights[i][j]
                    
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

        # 4. 競合学習に基づく厳格なシナプス更新
        for j in range(self.num_outputs):
            # 発火した「勝者ニューロン」のみがシナプスを更新する
            if output_spikes[j] == 1:
                for i in range(self.num_inputs):
                    if input_spikes[i] == 1:
                        self.weights[i][j] += self.A_plus
                    else:
                        self.weights[i][j] -= self.A_minus
                        
                # 厳格なSynaptic Scaling（重みの総量規制）
                # 緩やかな調整ではなく、直接総量を目標値に合わせる
                current_sum = sum(self.weights[i][j] for i in range(self.num_inputs))
                if current_sum > 0:
                    scale_factor = self.target_weight_sum / current_sum
                    for i in range(self.num_inputs):
                        self.weights[i][j] *= scale_factor
                        
                        # 最終的な結合荷重のクリッピング
                        if self.weights[i][j] > 1.0:
                            self.weights[i][j] = 1.0
                        elif self.weights[i][j] < 0.0:
                            self.weights[i][j] = 0.0

        return output_spikes, list(self.potentials)