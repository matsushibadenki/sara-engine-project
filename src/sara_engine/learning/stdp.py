# Directory Path: src/sara_engine/learning/stdp.py
# English Title: STDP and BCM Learning Layer with Forward-Forward and Burn-in
# Purpose/Content: Implementation of STDP and BCM-STDP hybrid layer for associative capacity and stable metaplasticity. Incorporates Forward-Forward (FF) algorithm and Burn-in strategy to achieve high accuracy without backpropagation or matrix operations. Multi-language support included.

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

        # 学習フェーズのみ重みを更新 (Forward-Forward対応 Reward-Modulated STDP)
        if reward != 0.0:
            active_set = set(active_inputs)
            for j in fired_indices:
                current_synapses = self.synapses[j]
                
                if reward > 0:
                    # 正例データ: Soft-bound LTP (上限に近づくほど増分が減る)
                    for i in list(current_synapses.keys()):
                        if i in active_set:
                            delta_w = self.A_plus * reward * (self.w_max - current_synapses[i]) / self.w_max
                            current_synapses[i] += delta_w
                        else:
                            # 報酬時のみ非アクティブ入力を抑圧
                            delta_w = self.A_minus * current_synapses[i]
                            current_synapses[i] -= delta_w
                            if current_synapses[i] < self.prune_threshold:
                                del current_synapses[i]
                else:
                    # 負例データ: ペナルティ (負例で発火した場合、結合を強く減衰させる)
                    penalty = abs(reward)
                    for i in list(current_synapses.keys()):
                        if i in active_set:
                            delta_w = self.A_plus * penalty * current_synapses[i] * 0.5
                            current_synapses[i] -= delta_w
                            if current_synapses[i] < self.prune_threshold:
                                del current_synapses[i]

                # シナプス新生 (正例の場合のみ許可)
                if reward > 0 and random.random() < 0.5 and active_inputs:
                    new_i = random.choice(active_inputs)
                    current_synapses[new_i] = current_synapses.get(new_i, 0.3) + 0.4

                # スケーリング (恒常性の維持)
                current_sum = sum(current_synapses.values())
                if current_sum > 0:
                    scale = self.target_weight_sum / current_sum
                    for i in current_synapses:
                        current_synapses[i] = min(self.w_max, current_synapses[i] * scale)

        return output_spikes, list(self.potentials)

    def process_sequence(self, sequence: list[list[int]], is_positive: bool = True, burn_in_steps: int = 2) -> list[list[int]]:
        """
        バーンイン戦略とForward-Forwardアルゴリズムを組み合わせた時系列一括処理。
        多言語のステータスログ出力機能を持たせることも可能。
        """
        output_sequence = []
        for step, input_spikes in enumerate(sequence):
            # バーンイン期間: 重み更新を行わず膜電位の安定化(空回し)のみ行う
            if step < burn_in_steps:
                current_reward = 0.0
            else:
                # Forward-Forward: 正例は+1.0、負例は-1.0として層をローカルに最適化
                current_reward = 1.0 if is_positive else -1.0
                
            out_spikes, _ = self.process_step(input_spikes, reward=current_reward)
            output_sequence.append(out_spikes)
            
        return output_sequence

    def forward(self, input_spike_indices: list[int], learning: bool = True) -> list[int]:
        """上位クラスからの呼び出し"""
        input_spikes = [0] * self.num_inputs
        for idx in input_spike_indices:
            if 0 <= idx < self.num_inputs:
                input_spikes[idx] = 1

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

    def get_ff_status_message(self, lang: str = "en") -> str:
        """多言語対応: Forward-ForwardとBurn-inの状態メッセージ"""
        messages = {
            "en": "Ready for Forward-Forward learning with Burn-in active.",
            "ja": "バーンインが有効なForward-Forward学習の準備が完了しました。",
            "fr": "Prêt pour l'apprentissage Forward-Forward avec Burn-in actif."
        }
        return messages.get(lang, messages["en"])


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


class BCMSTDPLayer(STDPLayer):
    """
    BCM-STDP Hybrid Layer
    Incorporates Bienenstock-Cooper-Munro (BCM) theory with STDP.
    This provides a sliding threshold for metaplasticity, ensuring stable learning
    and heterosynaptic LTD without needing error backpropagation.
    """
    def __init__(self, num_inputs: int, num_outputs: int, threshold: float = 0.5):
        super().__init__(num_inputs, num_outputs, threshold)
        self.y_traces = [0.0] * num_outputs
        self.theta_m = [0.1] * num_outputs # Sliding threshold for BCM
        self.bcm_learning_rate = 0.05
        self.trace_decay = 0.8
        self.theta_decay = 0.99 

    def get_status_message(self, lang: str = "en") -> str:
        """多言語ステータス取得機能"""
        messages = {
            "en": "BCM-STDP Layer initialized. Sliding threshold active for metaplasticity.",
            "ja": "BCM-STDPレイヤーが初期化されました。メタ可塑性のためのスライディング閾値が有効です。",
            "fr": "Couche BCM-STDP initialisée. Seuil glissant actif pour la métaplasticité."
        }
        return messages.get(lang, messages["en"])

    def process_step(self, input_spikes: list[int], reward: float = 1.0, boost: bool = False) -> tuple[list[int], list[float]]:
        output_spikes = [0] * self.num_outputs
        active_inputs = [i for i, s in enumerate(input_spikes) if s == 1]
        
        gain = 8.0 if boost else 1.0

        for j in range(self.num_outputs):
            self.potentials[j] *= self.leak_rate
            for i in active_inputs:
                if i in self.synapses[j]:
                    self.potentials[j] += self.synapses[j][i] * gain

        fired_indices = []
        for j in range(self.num_outputs):
            if self.potentials[j] >= self.thresholds[j]:
                output_spikes[j] = 1
                fired_indices.append(j)
                self.potentials[j] = 0.0 
                self.thresholds[j] += self.theta_plus
            
            # 発火履歴トレースの更新とBCMスライディング閾値(E[y^2])の計算
            self.y_traces[j] = self.y_traces[j] * self.trace_decay + output_spikes[j]
            self.theta_m[j] = self.theta_m[j] * self.theta_decay + (self.y_traces[j] ** 2) * (1.0 - self.theta_decay)

        for j in range(self.num_outputs):
            self.thresholds[j] += (self.base_threshold - self.thresholds[j]) * self.theta_decay

        if reward != 0.0:
            active_set = set(active_inputs)
            for j in range(self.num_outputs):
                current_synapses = self.synapses[j]
                y = self.y_traces[j]
                
                # BCM変調係数: y > theta_mで増強、y < theta_mで抑圧
                bcm_factor = y * (y - self.theta_m[j])
                
                if reward > 0:
                    for i in list(current_synapses.keys()):
                        if i in active_set:
                            delta_w = self.bcm_learning_rate * reward * bcm_factor
                            current_synapses[i] += delta_w
                        else:
                            if output_spikes[j]:
                                # ヘテロシナプティックLTD
                                current_synapses[i] -= self.A_minus * current_synapses[i]
                                
                        if current_synapses[i] < self.prune_threshold:
                            del current_synapses[i]
                        elif current_synapses[i] > self.w_max:
                            current_synapses[i] = self.w_max

                    if output_spikes[j] and random.random() < 0.5 and active_inputs:
                        new_i = random.choice(active_inputs)
                        current_synapses[new_i] = current_synapses.get(new_i, 0.3) + 0.4
                else:
                    # BCM層における負例ペナルティ処理
                    penalty = abs(reward)
                    for i in list(current_synapses.keys()):
                        if i in active_set and output_spikes[j]:
                            delta_w = self.bcm_learning_rate * penalty * current_synapses[i]
                            current_synapses[i] -= delta_w
                            if current_synapses[i] < self.prune_threshold:
                                del current_synapses[i]

                current_sum = sum(current_synapses.values())
                if current_sum > 0:
                    scale = self.target_weight_sum / current_sum
                    for i in current_synapses:
                        current_synapses[i] = min(self.w_max, current_synapses[i] * scale)

        return output_spikes, list(self.potentials)