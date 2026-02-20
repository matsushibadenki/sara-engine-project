# path: src/sara_engine/memory/million_token_snn.py
# title: 100万トークン対応イベント駆動型SNNメモリ(ハブ抑制・長距離結合版)
"""
{
    "//": "目的: 助詞などの高頻度単語が引き起こすスパイク発散（ハブ問題）をシナプススケーリングによって抑制し、時間窓の拡大による直接結合で正確な連想を実現する。"
}
"""

import math
from collections import defaultdict, deque
import time

class LIFNeuron:
    """
    多様な時定数（減衰率）を持つ漏れ積分発火（LIF）ニューロン。
    遅延評価（Lazy Evaluation）を用いて、毎ステップのループ計算を排除。
    """
    def __init__(self, neuron_id: int, is_slow: bool = False):
        self.id = neuron_id
        self.v = 0.0
        self.leak_rate = 0.9999 if is_slow else 0.7 
        self.threshold = 1.0
        self.last_spike_time = -1.0
        self.last_update_time = 0.0

    def update_and_get_voltage(self, current_time: float) -> float:
        """現在の時刻に基づく膜電位の遅延評価"""
        if self.v > 0:
            dt = current_time - self.last_update_time
            if dt > 0:
                self.v *= (self.leak_rate ** dt)
                if self.v < 0.001:
                    self.v = 0.0
        self.last_update_time = current_time
        return self.v

    def add_stimulus(self, amount: float, current_time: float) -> bool:
        """刺激を受け取り、閾値を超えれば発火(True)を返す"""
        self.update_and_get_voltage(current_time)
        self.v += amount
        if self.v >= self.threshold:
            self.v = 0.0 
            self.last_spike_time = current_time
            return True
        return False


class DynamicSNNMemory:
    """
    行列演算・誤差逆伝播を使用しない完全イベント駆動型のSNN。
    """
    def __init__(self, vocab_size: int, sdr_size: int = 5):
        self.vocab_size = vocab_size
        self.sdr_size = sdr_size
        self.current_time = 0.0
        
        self.neurons = {}
        self.synapses = defaultdict(lambda: defaultdict(float))
        
        self.recent_spikes = deque()
        
        # STDPのパラメータ
        self.tau_plus = 2.0
        self.tau_minus = 3.0
        self.a_plus = 1.5  # 1回の学習で強固な記憶を作るために学習率を高めに設定
        self.a_minus = 0.2
        self.max_weight = 5.0

        self.token_to_neurons = {}
        self.neuron_to_tokens = defaultdict(list)

    def _get_or_create_neuron(self, neuron_id: int) -> LIFNeuron:
        if neuron_id not in self.neurons:
            is_slow = (neuron_id % 2 != 0)
            self.neurons[neuron_id] = LIFNeuron(neuron_id, is_slow=is_slow)
        return self.neurons[neuron_id]

    def _encode_token(self, token_id: int) -> list:
        if token_id not in self.token_to_neurons:
            active_neurons = []
            for i in range(self.sdr_size):
                n_id = (token_id * 1103515245 + i * 12345) % (self.vocab_size * 10)
                active_neurons.append(n_id)
            self.token_to_neurons[token_id] = active_neurons
            for n_id in active_neurons:
                self.neuron_to_tokens[n_id].append(token_id)
        return self.token_to_neurons[token_id]

    def _apply_stdp(self, pre_id: int, post_id: int, pre_time: float, post_time: float):
        dt = post_time - pre_time
        # 時間窓を <= 3.0 に広げ、助詞を飛び越えて名詞同士が直接結合できるようにする
        if 0 < dt <= 3.0:
            dw = self.a_plus * math.exp(-dt / self.tau_plus)
            self.synapses[pre_id][post_id] = min(self.max_weight, self.synapses[pre_id][post_id] + dw)
        elif -3.0 <= dt < 0:
            dw = -self.a_minus * math.exp(dt / self.tau_minus)
            self.synapses[pre_id][post_id] = max(0.0, self.synapses[pre_id][post_id] + dw)

    def process_sequence(self, sequence: list, is_training: bool = True) -> list:
        output_tokens = []
        
        if is_training:
            for token_id in sequence:
                self.current_time += 1.0
                active_neurons = self._encode_token(token_id)
                
                for current_id in active_neurons:
                    neuron = self._get_or_create_neuron(current_id)
                    neuron.add_stimulus(2.0, self.current_time) 
                    
                    for prev_id, pre_time in self.recent_spikes:
                        if prev_id != current_id:
                            self._apply_stdp(prev_id, current_id, pre_time, self.current_time)
                    
                    self.recent_spikes.append((current_id, self.current_time))
                    
                while self.recent_spikes and self.current_time - self.recent_spikes[0][1] > 3.0:
                    self.recent_spikes.popleft()
            return []
            
        else:
            spiked_in_this_step = {}
            
            for token_id in sequence:
                self.current_time += 1.0
                active_neurons = self._encode_token(token_id)
                event_queue = deque([(n_id, 2.0, 0) for n_id in active_neurons])

                while event_queue:
                    current_id, stimulus, depth = event_queue.popleft()
                    
                    # 探索の深さは2に抑え、遠すぎるノイズを拾わないようにする
                    if depth > 2:
                        continue

                    neuron = self._get_or_create_neuron(current_id)
                    
                    if neuron.add_stimulus(stimulus, self.current_time):
                        if current_id not in spiked_in_this_step or depth < spiked_in_this_step[current_id]:
                            spiked_in_this_step[current_id] = depth

                        # シナプススケーリング（ハブ抑制）の適用
                        # 結合先が多すぎるニューロン（助詞などの高頻度単語）からの信号伝達を大きく減衰させる
                        fan_out = len(self.synapses[current_id])
                        hub_penalty = 1.0
                        if fan_out > 5:
                            hub_penalty = 5.0 / fan_out

                        for target_id, weight in self.synapses[current_id].items():
                            adjusted_weight = weight * hub_penalty
                            # 減衰後の刺激が閾値(0.5)を超える強い結合（名詞の直接結合）だけを伝播させる
                            if adjusted_weight > 0.5:
                                event_queue.append((target_id, adjusted_weight, depth + 1))

            if spiked_in_this_step:
                token_scores = defaultdict(float)
                for n_id, depth in spiked_in_this_step.items():
                    weight = 1.0 / (depth + 1.0)
                    for t_id in self.neuron_to_tokens.get(n_id, []):
                        token_scores[t_id] += weight
                
                input_set = set(sequence)
                sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
                
                for t_id, score in sorted_tokens:
                    if t_id not in input_set:
                        output_tokens.append(t_id)
                        if len(output_tokens) >= 3:
                            break

            return output_tokens

if __name__ == "__main__":
    print("=== 100万トークン対応 イベント駆動SNN 初期化 ===")
    snn_memory = DynamicSNNMemory(vocab_size=10000, sdr_size=3)
    
    training_sequence = [10, 20, 30, 40]
    
    print("1. コンテキストの学習を開始 (One-shot学習)...")
    start_time = time.time()
    snn_memory.process_sequence(training_sequence, is_training=True)
    print(f"学習完了: {time.time() - start_time:.4f}秒")

    print("\n2. 連想アテンションによる推論テスト...")
    test_sequence = [10]
    predicted = snn_memory.process_sequence(test_sequence, is_training=False)
    
    print(f"入力トークンID: {test_sequence}")
    print(f"推論された次のトークンID群: {predicted}")
    if 20 in predicted:
        print("成功: 行列演算なしのSTDPネットワークで正しく連想しました。")