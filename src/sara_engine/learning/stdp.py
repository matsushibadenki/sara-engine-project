_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/learning/stdp.py",
    "//": "ファイルの日本語タイトル: STDP（スパイクタイミング依存可塑性）学習レイヤー",
    "//": "ファイルの目的や内容: 想起フラグの確実な伝播と、物理的な膜電位加算による想起能力の最大化。Soft-bound STDPとReward-Modulated STDPの導入による学習精度と安定性の向上。"
}

import random

class STDPLayer:
    def __init__(self, num_inputs: int, num_outputs: int, threshold: float = 0.5):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.synapses = []
        for _ in range(num_outputs):
            connections = {}
            for i in range(num_inputs):
                if random.random() < 0.7: # 結線密度を70%まで引き上げ
                    connections[i] = random.uniform(0.1, 0.6)
            self.synapses.append(connections)

        self.A_plus = 0.5 # 学習効率を極大化
        self.A_minus = 0.05 # Soft-bound用に調整
        self.potentials = [0.0] * num_outputs
        self.leak_rate = 0.999
        self.base_threshold = threshold
        self.thresholds = [threshold] * num_outputs
        self.theta_plus = 0.01 # 発火後の感度低下を最小限に
        self.theta_decay = 0.3
        self.target_weight_sum = 10.0
        self.prune_threshold = 0.0001
        self.w_max = 4.0 # Soft-bound用の上限重み

    def process_step(self, input_spikes: list[int], reward: float = 1.0, boost: bool = False) -> tuple[list[int], list[float]]:
        output_spikes = [0] * self.num_outputs
        active_inputs = [i for i, s in enumerate(input_spikes) if s == 1]
        
        gain = 8.0 if boost else 1.0

        for j in range(self.num_outputs):
            self.potentials[j] *= self.leak_rate
            for i in active_inputs:
                if i in self.synapses[j]:
                    # 膜電位への直接加算を強化
                    self.potentials[j] += self.synapses[j][i] * gain

        fired_indices = []
        for j in range(self.num_outputs):
            if self.potentials[j] >= self.thresholds[j]:
                output_spikes[j] = 1
                fired_indices.append(j)
                self.potentials[j] = 0.0 # 発火後リセット
                self.thresholds[j] += self.theta_plus

        for j in range(self.num_outputs):
            self.thresholds[j] += (self.base_threshold - self.thresholds[j]) * self.theta_decay

        # 学習フェーズのみ重みを更新 (Reward-Modulated Soft-bound STDP)
        if reward > 0:
            active_set = set(active_inputs)
            for j in fired_indices:
                current_synapses = self.synapses[j]
                for i in list(current_synapses.keys()):
                    if i in active_set:
                        # Soft-bound LTP: 上限(w_max)に近づくほど増分が減る
                        delta_w = self.A_plus * reward * (self.w_max - current_synapses[i]) / self.w_max
                        current_synapses[i] += delta_w
                    else:
                        # Soft-bound LTD: 現在の重みに比例して減衰(Rewardは無視して定常的な忘却を促す)
                        delta_w = self.A_minus * current_synapses[i]
                        current_synapses[i] -= delta_w
                        
                        # 閾値を下回ったシナプスを完全に削除（刈り込み）
                        if current_synapses[i] < self.prune_threshold:
                            del current_synapses[i]
                
                # シナプス新生
                if random.random() < 0.5 and active_inputs:
                    new_i = random.choice(active_inputs)
                    current_synapses[new_i] = current_synapses.get(new_i, 0.3) + 0.4

                # スケーリング (恒常性の維持)
                current_sum = sum(current_synapses.values())
                if current_sum > 0:
                    scale = self.target_weight_sum / current_sum
                    for i in current_synapses:
                        current_synapses[i] = min(self.w_max, current_synapses[i] * scale)

        return output_spikes, list(self.potentials)

    def forward(self, input_spike_indices: list[int], learning: bool = True) -> list[int]:
        """上位クラスからの呼び出し"""
        input_spikes = [0] * self.num_inputs
        for idx in input_spike_indices:
            if 0 <= idx < self.num_inputs:
                input_spikes[idx] = 1

        # 明示的にlearningフラグを内部処理に分配
        output_spikes, _ = self.process_step(
            input_spikes, 
            reward=(1.0 if learning else 0.0), 
            boost=(not learning)
        )
        return [i for i, s in enumerate(output_spikes) if s == 1]

    def reset_state(self):
        """膜電位のみをリセットし、重みは絶対に保持する"""
        self.potentials = [0.0] * self.num_outputs
        self.thresholds = [self.base_threshold] * self.num_outputs


class STDPPretrainer:
    """
    コーパスなどのシーケンシャルデータから、単語/トークン間の遷移確率を
    Spike-Timing Dependent Plasticity (STDP) によって事前学習するクラス。
    行列演算・誤差逆伝播を一切使わず、純粋な辞書（スパース結合）でシナプスを形成する。
    """

    def __init__(self, window_size: int = 3, a_plus: float = 1.0, a_minus: float = 0.2):
        self.window_size = window_size
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = 5.0 # 事前学習における最大重み

    def pretrain(self, model, corpus: list[str]):
        if not hasattr(model, "synapses"):
            model.synapses = {}

        for text in corpus:
            tokens = text.split()
            sdr_sequence = []

            # 各トークンをSDR（発火インデックスのリスト）に変換
            for token in tokens:
                sdr = []
                if hasattr(model, "encoder"):
                    try:
                        if hasattr(model.encoder, "encode"):
                            sdr = model.encoder.encode(token)
                    except Exception:
                        pass

                # エンコード失敗時やテスト用のフォールバック（疑似発火）
                if not sdr:
                    sdr = [hash(token + str(i)) % 1024 for i in range(10)]

                sdr_sequence.append(sdr)

            # 時系列STDPの適用
            for i in range(len(sdr_sequence)):
                pre_sdr = sdr_sequence[i]
                for pre_idx in pre_sdr:
                    if pre_idx not in model.synapses:
                        model.synapses[pre_idx] = {}

                    # 1. 未来の発火に対するLTP (Causal) + Soft-bound
                    for j in range(1, self.window_size + 1):
                        if i + j < len(sdr_sequence):
                            post_sdr = sdr_sequence[i + j]
                            # 距離が遠いほど重みの増加を減衰させる
                            base_update = self.a_plus / j
                            for post_idx in post_sdr:
                                current_w = model.synapses[pre_idx].get(post_idx, 0.0)
                                # Soft-boundによる更新量の計算
                                update = base_update * (self.w_max - current_w) / self.w_max
                                model.synapses[pre_idx][post_idx] = min(self.w_max, current_w + update)

                    # 2. 過去の発火に対するLTD (Anti-causal) + プルーニング
                    for j in range(1, self.window_size + 1):
                        if i - j >= 0:
                            past_sdr = sdr_sequence[i - j]
                            base_penalty = self.a_minus / j
                            for past_idx in past_sdr:
                                if past_idx in model.synapses[pre_idx]:
                                    current_w = model.synapses[pre_idx][past_idx]
                                    # 現在の重みに比例して減衰
                                    new_w = current_w - (base_penalty * current_w)
                                    if new_w < 0.01:
                                        del model.synapses[pre_idx][past_idx]
                                    else:
                                        model.synapses[pre_idx][past_idx] = new_w