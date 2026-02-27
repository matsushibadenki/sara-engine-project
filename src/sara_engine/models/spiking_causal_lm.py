_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_causal_lm.py",
    "//": "ファイルの日本語タイトル: スパイキング因果言語モデル",
    "//": "ファイルの目的や内容: BPE導入によるサブワード分割に伴い、文脈保持のための軸索遅延（max_delay）を10に拡張。また、フレーズ単位でのループを防ぐため不応期の範囲を拡大。"
}

import json
import random
import math
from typing import List, Dict, Tuple, Optional
from sara_engine.core.transformer import SpikeTransformerModel

class SpikingCausalLM:
    def __init__(self, vocab_size: int, embed_dim: int = 1024, hidden_dim: int = 2048, 
                 num_layers: int = 2, use_lif: bool = True):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # BPEトークンの長さに合わせて過去10ステップ前までの遅延線を保持
        self.max_delay = 10 
        
        self.transformer = SpikeTransformerModel(
            num_layers=num_layers, 
            embed_dim=embed_dim, 
            hidden_dim=hidden_dim, 
            use_lif=use_lif
        )
        self.token_to_sdr: Dict[int, List[int]] = {}
        
        # weights[delay][spike_id][token_id] 
        self.weights: List[Dict[int, Dict[int, float]]] = [{} for _ in range(self.max_delay + 1)]

    def _get_sdr_for_token(self, token_id: int, sparsity: float = 0.05) -> List[int]:
        if token_id not in self.token_to_sdr:
            random.seed(token_id)
            num_spikes = max(1, int(self.embed_dim * sparsity))
            self.token_to_sdr[token_id] = random.sample(range(self.embed_dim), num_spikes)
            random.seed()
        return self.token_to_sdr[token_id]

    def reset_context(self):
        """膜電位と状態をリセット"""
        self.transformer.reset_state()

    def train_step(self, sequence: List[int], learning_rate: float = 0.5):
        """
        破壊的忘却を防ぐための純粋なヘッブ則（加算のみのSTDP学習）
        """
        self.reset_context()
        spike_history: List[List[int]] = []
        
        for i in range(len(sequence) - 1):
            curr, nxt = sequence[i], sequence[i + 1]
            input_spikes = self._get_sdr_for_token(curr)
            
            # LSMとして動作 (learning=False)
            output_spikes = self.transformer.forward(input_spikes, learning=False)
            
            # 感覚スパイク(SDR)と文脈スパイク(LSM)を統合
            offset_context_spikes = [s + self.embed_dim for s in output_spikes]
            combined_spikes = input_spikes + offset_context_spikes
            
            # 遅延バッファの更新
            spike_history.insert(0, combined_spikes)
            if len(spike_history) > self.max_delay + 1:
                spike_history.pop()
                
            # 各遅延線ごとにシナプスを強化
            for delay, active_spikes in enumerate(spike_history):
                # 古い記憶ほど学習率を緩やかに落とす
                eff_lr = learning_rate * (1.0 - delay * 0.08) 
                if eff_lr <= 0: continue
                
                for s in active_spikes:
                    if s not in self.weights[delay]:
                        self.weights[delay][s] = {}
                        
                    # Oja近似によるシナプス強化 (Soft-bound at 1.0)
                    old_w = self.weights[delay][s].get(nxt, 0.0)
                    self.weights[delay][s][nxt] = old_w + eff_lr * (1.0 - old_w)

    def generate(self, prompt_tokens: List[int], max_new_tokens: int = 15, temperature: float = 1.0) -> List[int]:
        self.reset_context()
        generated = []
        spike_history: List[List[int]] = []
        
        # ハブ単語へのペナルティ事前計算 (Inverse Fan-in)
        token_fan_in: Dict[int, float] = {}
        for delay_dict in self.weights:
            for s_dict in delay_dict.values():
                for t, w in s_dict.items():
                    token_fan_in[t] = token_fan_in.get(t, 0.0) + w

        # 1. プロンプトの流し込みと遅延線の構築
        for t in prompt_tokens[:-1]:
            input_spikes = self._get_sdr_for_token(t)
            output_spikes = self.transformer.forward(input_spikes, learning=False)
            combined_spikes = input_spikes + [s + self.embed_dim for s in output_spikes]
            
            spike_history.insert(0, combined_spikes)
            if len(spike_history) > self.max_delay + 1:
                spike_history.pop()
                
        current_token = prompt_tokens[-1]
        
        # 2. 生成ループ
        for _ in range(max_new_tokens):
            input_spikes = self._get_sdr_for_token(current_token)
            output_spikes = self.transformer.forward(input_spikes, learning=False)
            combined_spikes = input_spikes + [s + self.embed_dim for s in output_spikes]
            
            spike_history.insert(0, combined_spikes)
            if len(spike_history) > self.max_delay + 1:
                spike_history.pop()
                
            token_potentials: Dict[int, float] = {}
            
            # 時間的に一致したスパイクの電位を全て統合 (Polychronization)
            for delay, active_spikes in enumerate(spike_history):
                for s in active_spikes:
                    if s in self.weights[delay]:
                        for t_id, weight in self.weights[delay][s].items():
                            token_potentials[t_id] = token_potentials.get(t_id, 0.0) + weight

            if not token_potentials:
                break
                
            # Inverse Fan-in Penaltyの適用（Hubトークンの過剰発火を抑制）
            for t_id in token_potentials:
                hub_factor = token_fan_in.get(t_id, 1.0)
                if hub_factor > 1.0:
                    # ペナルティをやや強める
                    token_potentials[t_id] /= (math.sqrt(hub_factor) * 1.2)
            
            # 絶対不応期: BPEに合わせて直近5トークンの出力を強力に抑制
            forbidden = set(generated[-5:] + [current_token])
            for f_id in forbidden:
                if f_id in token_potentials:
                    token_potentials[f_id] *= 0.001

            sorted_candidates = sorted(token_potentials.items(), key=lambda x: x[1], reverse=True)
            
            if not sorted_candidates:
                break

            # サンプリング
            if temperature <= 0.1:
                next_token = sorted_candidates[0][0] # Greedy
            else:
                top_k = min(5, len(sorted_candidates))
                candidates = sorted_candidates[:top_k]
                total_pot = sum(pow(p, 1.0/temperature) for _, p in candidates)
                r = random.uniform(0, total_pot)
                
                next_token = candidates[0][0]
                cumulative = 0.0
                for t_id, pot in candidates:
                    cumulative += pow(pot, 1.0/temperature)
                    if r <= cumulative:
                        next_token = t_id
                        break
            
            if token_potentials.get(next_token, 0) < 0.05:
                break
                
            generated.append(next_token)
            current_token = next_token
            
        return generated

    def save_pretrained(self, filepath: str):
        serializable_weights = []
        for delay_dict in self.weights:
            serializable_weights.append({
                str(k): {str(tk): v for tk, v in tv.items()} for k, tv in delay_dict.items()
            })
            
        state = {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "max_delay": self.max_delay,
            "transformer": self.transformer.state_dict(),
            "weights": serializable_weights
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_pretrained(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        self.vocab_size = state["vocab_size"]
        self.embed_dim = state["embed_dim"]
        self.max_delay = state.get("max_delay", 10)
        self.transformer.load_state_dict(state["transformer"])
        
        self.weights = []
        for delay_dict in state.get("weights", []):
            self.weights.append({
                int(k): {int(tk): float(v) for tk, v in tv.items()} for k, tv in delay_dict.items()
            })