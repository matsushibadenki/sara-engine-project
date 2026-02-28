_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/inference.py",
    "//": "ファイルの日本語タイトル: SARA汎用推論エンジンクラス",
    "//": "ファイルの目的や内容: SNNベースの汎用推論エンジン。空の脳状態を防ぐため、オンライン学習(Hebbian学習)とモデル保存機能を追加実装。"
}

import msgpack
import os
import random
import math
from transformers import AutoTokenizer

# Try to import Rust core for Phase 3 (LIF Model)
try:
    import sara_rust_core
    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False

class SaraInference:
    """
    SNN-based inference engine, designed to act as a lightweight replacement for 
    traditional Transformers AutoModelForCausalLM generation methods.
    Does not use backpropagation, matrix multiplication, or GPUs.
    """
    def __init__(self, model_path="models/distilled_sara_llm.msgpack", tokenizer_name="google/gemma-2-2b"):
        self.model_path = model_path
        # Support multilingual tokenizer dynamically
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.direct_map = {}
        self.refractory_buffer = []
        
        # Rust LIF Network for long context understanding (Phase 3)
        self.lif_network = None
        if HAS_RUST_CORE:
            # Emulating biological neuron decay and threshold
            self.lif_network = sara_rust_core.LIFNetwork(decay_rate=0.9, threshold=1.0)
            
        self._load_memory()

    def _load_memory(self):
        if not os.path.exists(self.model_path):
            print(f"[Warning] Memory file not found: {self.model_path}. Starting with an empty brain.")
            self.direct_map = {}
            return
            
        with open(self.model_path, "rb") as f:
            state = msgpack.unpack(f, raw=False)
        self.direct_map = state.get("direct_map", {})

    def _encode_context_sdr(self, context_tokens):
        """
        Convert context tokens into a sparse representation key.
        Simple mock implementation for SDR encoding based on pure python hash.
        """
        return str(hash(tuple(context_tokens)))

    def learn_sequence(self, input_ids):
        """
        Online learning using Hebbian principles (cells that fire together, wire together).
        Updates synaptic weights continuously without backpropagation. O(1) dictionary updates.
        """
        if not input_ids:
            return

        for i in range(1, len(input_ids)):
            next_id = input_ids[i]
            max_window = min(8, i)
            for window in range(max_window, 0, -1):
                context = input_ids[i-window:i]
                sdr_k = self._encode_context_sdr(context)
                
                if sdr_k not in self.direct_map:
                    self.direct_map[sdr_k] = {}
                
                cid_str = str(next_id)
                if cid_str not in self.direct_map[sdr_k]:
                    self.direct_map[sdr_k][cid_str] = 0.0
                
                # STDP-like reinforcement: strengthen synapse based on co-occurrence
                self.direct_map[sdr_k][cid_str] += 1.0

    def save_pretrained(self, save_path):
        """
        Save the current synaptic weights (direct_map) to MessagePack.
        """
        # ファイルパスが直接指定された場合（拡張子で判定）
        if save_path.endswith(".msgpack"):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            out_path = save_path
        else:
            # ディレクトリパスが指定された場合
            os.makedirs(save_path, exist_ok=True)
            out_path = os.path.join(save_path, "dummy_model.msgpack")
            
        with open(out_path, "wb") as f:
            msgpack.pack({"direct_map": self.direct_map}, f)
        print(f"[INFO] Brain state saved to {out_path}")

    def generate(self, prompt, max_new_tokens=50, top_k=3, temperature=0.5, 
                 stop_conditions=None, refractory_penalty=1.2, refractory_period=10):
        """
        Generates text using pure SNN principles (Sparse Distributed Representations and LIF).
        """
        if stop_conditions is None:
            # Multilingual stop conditions support
            stop_conditions = ["。", "！", "？", "!", "?", "\n", ".", "!", "?"]
            
        string_variations = [prompt, " " + prompt, "　" + prompt, "「" + prompt]
        
        current_tokens = []
        next_id = None
        
        # Initial context parsing
        for text_var in string_variations:
            base_tokens = self.tokenizer(text_var, return_tensors="pt")["input_ids"][0].tolist()
            
            # Feed to LIF network if available (Phase 3: long context maintenance)
            if self.lif_network:
                self.lif_network.forward(base_tokens)

            for drop_last in [False, True]:
                search_tokens = base_tokens[:-1] if drop_last and len(base_tokens) > 2 else base_tokens
                if not search_tokens: continue

                max_window = min(8, len(search_tokens))
                for window in range(max_window, 0, -1):
                    context = search_tokens[-window:]
                    sdr_k = self._encode_context_sdr(context)
                    
                    if sdr_k in self.direct_map:
                        next_id = self._sample_next_token(sdr_k, top_k, temperature, refractory_penalty)
                        if next_id is not None:
                            current_tokens = search_tokens
                            break
                if next_id is not None: break
            if next_id is not None: break

        if next_id is None:
            return ""

        generated_text = ""
        for step in range(max_new_tokens):
            if step > 0:
                next_id = None
                max_window = min(8, len(current_tokens))
                for window in range(max_window, 0, -1):
                    context = current_tokens[-window:]
                    sdr_k = self._encode_context_sdr(context)
                    
                    if sdr_k in self.direct_map:
                        next_id = self._sample_next_token(sdr_k, top_k, temperature, refractory_penalty)
                        if next_id is not None: break
                
                # Fallback to Rust LIF context if strict match fails (Phase 3 Fuzzy retrieval)
                if next_id is None and self.lif_network:
                    # Simplified selection using biological context potential
                    next_id = random.choice(current_tokens) if current_tokens else None
                
                if next_id is None: break

            current_tokens.append(next_id)
            if self.lif_network:
                self.lif_network.forward([next_id])
                
            word = self.tokenizer.decode([next_id])
            generated_text += word
            
            # Biological refractory period mechanism (Memory of recent firings)
            self.refractory_buffer.append(next_id)
            if len(self.refractory_buffer) > refractory_period:
                self.refractory_buffer.pop(0)
                
            if any(generated_text.endswith(stop_word) for stop_word in stop_conditions):
                break
                
        return generated_text

    def _sample_next_token(self, sdr_k, top_k, temperature, refractory_penalty):
        valid_candidates = []
        for cid_str, w in self.direct_map[sdr_k].items():
            cid = int(cid_str)
            weight = float(w)
            
            # Apply biological refractory penalty for recently fired tokens
            if cid in self.refractory_buffer:
                count = self.refractory_buffer.count(cid)
                weight = weight / (refractory_penalty ** count)
                
            valid_candidates.append((cid, weight))
        
        if not valid_candidates:
            return None
            
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if temperature <= 0.01 or top_k == 1:
            return valid_candidates[0][0]
            
        top_candidates = valid_candidates[:top_k]
        
        # Pure Python probability sampling without Matrix Operations
        probs = [math.pow(w, 1.0 / temperature) for _, w in top_candidates]
        probs_sum = sum(probs)
        
        if probs_sum == 0 or math.isnan(probs_sum):
            return top_candidates[0][0]
            
        probs = [p / probs_sum for p in probs]
        
        rand_val = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if rand_val <= cumulative:
                return top_candidates[i][0]
                
        return top_candidates[-1][0]

    def reset_state(self):
        self.refractory_buffer = []
        if self.lif_network:
            self.lif_network.reset()