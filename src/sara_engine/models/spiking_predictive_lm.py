# ディレクトリパス: src/sara_engine/models/spiking_predictive_lm.py
# ファイルの日本語タイトル: 予測符号化ベースのスパイク言語モデル
# ファイルの目的や内容: 階層モデルと予測符号化層を統合。最新のPredictiveCodingManagerを使用。直前1トークンだけでなく過去複数トークンのSDRを文脈として蓄積（context_sdr）することで、マルコフ性を超えた長距離依存性（"You are a" -> "dog"）の自己組織化と自己回帰生成を実現する。

import random
from typing import List, Dict, Any, Set
from .hierarchical_snn import HierarchicalSNN
from ..nn.module import SNNModule
from ..learning.predictive_coding import PredictiveCodingManager

class SpikingPredictiveLM(SNNModule):
    def __init__(self, vocab_size: int, layer_configs: List[Dict[str, Any]], max_delay: int = 10, learning_rate: float = 0.2, predict_threshold: float = 0.5):
        super().__init__()
        print("\n[DEBUG] 長距離文脈(Long Context)対応の SpikingPredictiveLM が読み込まれました！")
        
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.predict_threshold = predict_threshold

        self.random_gen = random.Random(42)
        self.token_to_sdr: Dict[int, List[int]] = {}

        self.encoder = HierarchicalSNN(layer_configs=layer_configs)
        self.predictive_manager = PredictiveCodingManager(learning_rate=learning_rate)
        
        self.causal_weights: List[Dict[int, float]] = [{} for _ in range(10000)]
        
        # 直前1ステップだけでなく、長距離の文脈を蓄積するリスト
        self.context_sdr: List[int] = []
        self.prev_sdr: List[int] = []

        self.decoder_weights: Dict[int, Dict[int, float]] = {}

    def forward(self, token_ids: List[int], learning: bool = True) -> List[int]:
        sdr_spikes = []
        for t_id in token_ids:
            if t_id not in self.token_to_sdr:
                low_level = self.random_gen.sample(range(128), 16)
                start_id = 1000 + t_id * 200
                high_level = list(range(start_id, start_id + 200))
                self.token_to_sdr[t_id] = low_level + high_level
            sdr_spikes.extend(self.token_to_sdr[t_id])

        _ = self.encoder.forward(sdr_spikes, learning=False)
        pure_sdr_spikes = [s for s in sdr_spikes if s >= 1000]

        potentials: Dict[int, float] = {}
        if self.prev_sdr:
            for p_spike in self.prev_sdr:
                for target, weight in self.causal_weights[p_spike].items():
                    potentials[target] = potentials.get(target, 0.0) + weight

        predicted_spikes: Set[int] = {t for t, p in potentials.items() if p >= self.predict_threshold}
        
        actual_set = set(pure_sdr_spikes)
        surprise_spikes = list(actual_set - predicted_spikes)

        if learning and self.prev_sdr:
            self.predictive_manager.update_backward(
                backward_weights=self.causal_weights,
                prev_state_spikes=self.prev_sdr,
                current_in_spikes=pure_sdr_spikes,
                predicted_in_spikes=predicted_spikes,
                lr=self.learning_rate
            )

        # --- 長距離文脈(Long Context)の蓄積 ---
        self.context_sdr.extend(pure_sdr_spikes)
        # 過去3トークン分（1トークン200スパイク × 3 = 600）の文脈を保持
        if len(self.context_sdr) > 600:
            self.context_sdr = self.context_sdr[-600:]
            
        # 次の予測には、蓄積された長距離文脈を使用する
        self.prev_sdr = self.context_sdr

        if learning:
            for h_spike in pure_sdr_spikes:
                if h_spike not in self.decoder_weights:
                    self.decoder_weights[h_spike] = {}

                for t_id in list(self.decoder_weights[h_spike].keys()):
                    if t_id not in token_ids:
                        self.decoder_weights[h_spike][t_id] = max(
                            0.0, self.decoder_weights[h_spike][t_id] - 0.2)
                        if self.decoder_weights[h_spike][t_id] <= 0.0:
                            del self.decoder_weights[h_spike][t_id]

                for t_id in token_ids:
                    current_w = self.decoder_weights[h_spike].get(t_id, 0.0)
                    self.decoder_weights[h_spike][t_id] = min(
                        3.0, current_w + 1.0)

        return surprise_spikes

    def _predict_next_sdr(self) -> List[int]:
        if not self.prev_sdr:
            return []
        
        potentials: Dict[int, float] = {}
        for p_spike in self.prev_sdr:
            for target, weight in self.causal_weights[p_spike].items():
                potentials[target] = potentials.get(target, 0.0) + weight
                
        return [t for t, p in potentials.items() if p >= self.predict_threshold]

    def generate(self, prompt_tokens: List[int], max_length: int = 10) -> List[int]:
        generated_sequence = list(prompt_tokens)
        refractory_penalties: Dict[int, float] = {}

        for p_token in prompt_tokens:
            self.forward([p_token], learning=False)
            refractory_penalties[p_token] = 1000.0

        for _ in range(max_length):
            predicted_sdr = self._predict_next_sdr()
            if not predicted_sdr:
                break

            token_potentials: Dict[int, float] = {}
            for h_spike in predicted_sdr:
                if h_spike in self.decoder_weights:
                    for t_id, weight in self.decoder_weights[h_spike].items():
                        token_potentials[t_id] = token_potentials.get(t_id, 0.0) + weight

            for t_id in token_potentials.keys():
                if t_id in refractory_penalties:
                    token_potentials[t_id] -= refractory_penalties[t_id]

            valid_potentials = {k: v for k, v in token_potentials.items() if v > 0.0}
            if not valid_potentials:
                break

            next_token = max(valid_potentials.items(), key=lambda x: x[1])[0]
            generated_sequence.append(next_token)

            for t_id in list(refractory_penalties.keys()):
                refractory_penalties[t_id] *= 0.5
                if refractory_penalties[t_id] < 1.0:
                    del refractory_penalties[t_id]

            refractory_penalties[next_token] = 1000.0
            self.forward([next_token], learning=False)

        return generated_sequence

    def reset_state(self) -> None:
        self.encoder.reset_state()
        self.context_sdr = []
        self.prev_sdr = []
        super().reset_state()