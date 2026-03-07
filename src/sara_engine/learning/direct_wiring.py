# src/sara_engine/learning/direct_wiring.py
# 日本語タイトル: 直接シナプス結線型SNN学習モジュール
# 目的: コーパスからトークンの遷移確率を直接シナプス重みとして構築し、行列演算やBPを用いずに高速な学習とテキスト生成を行う。
# {
#     "//": "BP、行列演算、GPUを一切使用せず、Pythonの標準的な辞書を用いてスパースなシナプス結合を表現します。",
#     "//": "英語などのアルファベット言語でのループを防ぐため、ALIF（適応的閾値）によるホメオスタシスと、PMIベースの重み正規化を導入します。",
#     "//": "大量のコーパス学習を瞬時に完了させるため、シナプスの集計処理をRust側にオフロードします。"
# }

import os
import json
import math
from collections import defaultdict
from typing import Dict


class ALIFNeuron:
    def __init__(self, neuron_id: int, base_threshold: float = 1.0, decay: float = 0.5):
        self.neuron_id = neuron_id
        self.potential = 0.0
        self.base_threshold = base_threshold
        self.threshold = base_threshold
        self.decay = decay
        self.last_spike_time = -100

        # ALIF (Adaptive Leaky Integrate-and-Fire) パラメータ
        self.theta_plus = 2.0
        self.theta_decay = 0.8

    def integrate_and_fire(self, current: float, current_time: int) -> bool:
        self.threshold = self.base_threshold + \
            (self.threshold - self.base_threshold) * self.theta_decay

        if current_time - self.last_spike_time <= 1:
            current = 0.0

        self.potential = (self.potential * self.decay) + current
        if self.potential >= self.threshold:
            self.potential = 0.0
            self.last_spike_time = current_time
            self.threshold += self.theta_plus
            return True
        return False


class DirectWiringSNN:
    def __init__(self, vocab_size: int = 65536, context_window: int = 10):
        self.vocab_size = vocab_size
        self.context_window = context_window

        self.synapses: Dict[int, Dict[int, Dict[int, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float)))
        self.neurons: Dict[int, ALIFNeuron] = {}
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.next_id = 0

    def _get_neuron(self, token_id: int) -> ALIFNeuron:
        if token_id not in self.neurons:
            self.neurons[token_id] = ALIFNeuron(neuron_id=token_id, decay=0.3)
        return self.neurons[token_id]

    def _get_or_add_id(self, char: str) -> int:
        if char not in self.char_to_id:
            self.char_to_id[char] = self.next_id
            self.id_to_char[self.next_id] = char
            self.next_id += 1
        return self.char_to_id[char]

    def train_from_corpus(self, text_data: str):
        tokens = [self._get_or_add_id(c) for c in text_data]
        total_tokens = len(tokens)

        print(
            f"[INFO] Processing {total_tokens} characters for delay-line direct wiring...")

        # Rust拡張のロードを試みる (プロジェクトの構成に合わせて sara_engine からインポート)
        try:
            from .. import sara_rust_core
            print("[INFO] Utilizing Rust core for ultra-fast synaptic wiring...")

            # Rust側で全シナプス結線の集計とPMI正規化を高速実行
            rust_synapses = sara_rust_core.build_direct_synapses(
                tokens, self.context_window)

            self.synapses.clear()
            for delay_key, pre_dict in rust_synapses.items():
                delay = int(delay_key)
                for pre_key, post_dict in pre_dict.items():
                    pre = int(pre_key)
                    for post_key, weight in post_dict.items():
                        self.synapses[delay][pre][int(
                            post_key)] = float(weight)

            total_connections = sum(len(post_dict) for delay_dict in self.synapses.values(
            ) for post_dict in delay_dict.values())
            print(
                f"[INFO] Synaptic wiring completed via Rust. Established {total_connections} delay-line connections.")
            return

        except ImportError as e:
            print(
                f"[WARNING] Failed to load sara_rust_core: {e}. Did you forget to run `pip install -e .`?")
            print("[INFO] Falling back to standard Python implementation...")

        # フォールバック (Python単独実行時)
        co_occurrence = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float)))
        unigram_counts = defaultdict(int)

        for i in range(total_tokens):
            current_token = tokens[i]
            unigram_counts[current_token] += 1

            end_idx = min(i + self.context_window + 1, total_tokens)
            for j in range(i + 1, end_idx):
                delay = j - i
                next_token = tokens[j]
                co_occurrence[delay][current_token][next_token] += 1.0

        print("[INFO] Normalizing directional causal weights using PMI approach...")
        for delay, pre_dict in co_occurrence.items():
            for pre_token, posts in pre_dict.items():
                pre_count = unigram_counts[pre_token]
                for post_token, count in posts.items():
                    post_count = unigram_counts[post_token]
                    weight = count / math.sqrt(pre_count * post_count)
                    self.synapses[delay][pre_token][post_token] = weight

        total_connections = sum(len(post_dict) for delay_dict in self.synapses.values(
        ) for post_dict in delay_dict.values())
        print(
            f"[INFO] Synaptic wiring completed. Established {total_connections} delay-line connections.")

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        serializable_synapses = {}
        for delay, pre_dict in self.synapses.items():
            serializable_synapses[str(delay)] = {}
            for pre, post_dict in pre_dict.items():
                serializable_synapses[str(delay)][str(pre)] = {
                    str(post): w for post, w in post_dict.items()}

        model_data = {
            "synapses": serializable_synapses,
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
            "next_id": self.next_id,
            "context_window": self.context_window
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.synapses.clear()
        for delay_str, pre_dict in data.get("synapses", {}).items():
            delay = int(delay_str)
            for pre_str, post_dict in pre_dict.items():
                pre = int(pre_str)
                for post_str, w in post_dict.items():
                    self.synapses[delay][pre][int(post_str)] = float(w)

        self.char_to_id = data.get("char_to_id", {})
        self.id_to_char = {int(k): v for k, v in data.get(
            "id_to_char", {}).items()}
        self.next_id = data.get("next_id", 0)
        self.context_window = data.get("context_window", self.context_window)

        print(f"[INFO] Model loaded from {filepath}")

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        for neuron in self.neurons.values():
            neuron.potential = 0.0
            neuron.threshold = neuron.base_threshold
            neuron.last_spike_time = -100

        input_tokens = [self.char_to_id[c]
                        for c in prompt if c in self.char_to_id]
        if not input_tokens:
            return prompt

        generated_tokens = list(input_tokens)
        current_time = 0

        for idx, token in enumerate(input_tokens):
            neuron = self._get_neuron(token)
            neuron.last_spike_time = current_time - (len(input_tokens) - idx)
            neuron.threshold += neuron.theta_plus

        recent_spikes = input_tokens[-self.context_window:]

        for step in range(max_new_tokens):
            current_time += 1
            next_currents = defaultdict(float)

            for reversed_idx, pre_token in enumerate(reversed(recent_spikes)):
                delay = reversed_idx + 1

                if delay in self.synapses and pre_token in self.synapses[delay]:
                    for post_token, weight in self.synapses[delay][pre_token].items():
                        temporal_decay = 0.85 ** (delay - 1)
                        next_currents[post_token] += weight * temporal_decay

            new_active_spikes = []
            best_candidate = -1
            highest_potential = -1.0

            for target_token, current in next_currents.items():
                neuron = self._get_neuron(target_token)
                spiked = neuron.integrate_and_fire(current, current_time)

                if spiked:
                    new_active_spikes.append((target_token, neuron.potential))

                if neuron.potential > highest_potential:
                    highest_potential = neuron.potential
                    best_candidate = target_token

            if new_active_spikes:
                new_active_spikes.sort(key=lambda x: x[1], reverse=True)
                winner_token = new_active_spikes[0][0]
            elif best_candidate != -1:
                winner_token = best_candidate
            else:
                break

            winner_neuron = self._get_neuron(winner_token)
            winner_neuron.potential = 0.0
            winner_neuron.last_spike_time = current_time
            winner_neuron.threshold += winner_neuron.theta_plus

            generated_tokens.append(winner_token)

            recent_spikes.append(winner_token)
            if len(recent_spikes) > self.context_window:
                recent_spikes.pop(0)

        return "".join([self.id_to_char.get(tid, "") for tid in generated_tokens])
