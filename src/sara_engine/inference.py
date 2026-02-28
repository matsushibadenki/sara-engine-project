{
    "//": "ディレクトリパス: src/sara_engine/inference.py",
    "//": "ファイルの日本語タイトル: SARA汎用推論エンジンクラス",
    "//": "ファイルの目的や内容: SNNベースの汎用推論エンジン。Transformersライクな生成パラメータ(repetition_penalty等)をサポートし、生物学的な不応期をシミュレートする。"
}

import msgpack
import os
import numpy as np
from transformers import AutoTokenizer
from .models.spiking_llm import SpikingLLM

class SaraInference:
    def __init__(self, model_path="models/distilled_sara_llm.msgpack", sdr_size=8192, tokenizer_name="google/gemma-2-2b"):
        self.model_path = model_path
        # Support multilingual tokenizer dynamically
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.student = SpikingLLM(num_layers=2, sdr_size=sdr_size, vocab_size=256000)
        self.direct_map = {}
        self.refractory_buffer = []
        
        self._load_memory()

    def _load_memory(self):
        if not os.path.exists(self.model_path):
            print(f"[Warning] Memory file not found: {self.model_path}. Starting with an empty brain.")
            self.direct_map = {}
            return
            
        with open(self.model_path, "rb") as f:
            state = msgpack.unpack(f, raw=False)
        self.direct_map = state.get("direct_map", {})

    def generate(self, prompt, max_length=50, top_k=3, temperature=0.5, stop_conditions=None, repetition_penalty=1.2):
        if stop_conditions is None:
            # Multilingual stop conditions support
            stop_conditions = ["。", "！", "？", "!", "?", "\n", ".", "!", "?"]
            
        string_variations = [prompt, " " + prompt, "　" + prompt, "「" + prompt]
        
        current_tokens = []
        next_id = None
        
        for text_var in string_variations:
            base_tokens = self.tokenizer(text_var, return_tensors="pt")["input_ids"][0].tolist()
            
            for drop_last in [False, True]:
                search_tokens = base_tokens[:-1] if drop_last and len(base_tokens) > 2 else base_tokens
                if not search_tokens: continue

                max_window = min(8, len(search_tokens))
                for window in range(max_window, 0, -1):
                    context = search_tokens[-window:]
                    sdr_k = str(self.student._sdr_key(self.student._encode_to_sdr(context)))
                    
                    if sdr_k in self.direct_map:
                        next_id = self._sample_next_token(sdr_k, top_k, temperature, repetition_penalty)
                        if next_id is not None:
                            current_tokens = search_tokens
                            break
                if next_id is not None: break
            if next_id is not None: break

        if next_id is None:
            return ""

        generated_text = ""
        for step in range(max_length):
            if step > 0:
                next_id = None
                max_window = min(8, len(current_tokens))
                for window in range(max_window, 0, -1):
                    context = current_tokens[-window:]
                    sdr_k = str(self.student._sdr_key(self.student._encode_to_sdr(context)))
                    
                    if sdr_k in self.direct_map:
                        next_id = self._sample_next_token(sdr_k, top_k, temperature, repetition_penalty)
                        if next_id is not None: break
                
                if next_id is None: break

            current_tokens.append(next_id)
            word = self.tokenizer.decode([next_id])
            generated_text += word
            
            # Biological refractory period mechanism (Memory of recent firings)
            self.refractory_buffer.append(next_id)
            if len(self.refractory_buffer) > 10:
                self.refractory_buffer.pop(0)
                
            if any(generated_text.endswith(stop_word) for stop_word in stop_conditions):
                break
                
        return generated_text

    def _sample_next_token(self, sdr_k, top_k, temperature, repetition_penalty):
        valid_candidates = []
        for cid_str, w in self.direct_map[sdr_k].items():
            cid = int(cid_str)
            weight = w
            
            # Apply biological refractory penalty for recently fired tokens
            if cid in self.refractory_buffer:
                count = self.refractory_buffer.count(cid)
                weight = weight / (repetition_penalty ** count)
                
            valid_candidates.append((cid, weight))
        
        if not valid_candidates:
            return None
            
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if temperature <= 0.01 or top_k == 1:
            return valid_candidates[0][0]
            
        top_candidates = valid_candidates[:top_k]
        weights = np.array([w for _, w in top_candidates])
        
        probs = np.power(weights, 1.0 / temperature)
        probs_sum = probs.sum()
        
        if probs_sum == 0 or np.isnan(probs_sum):
            return top_candidates[0][0]
            
        probs /= probs_sum
        chosen_index = np.random.choice(len(top_candidates), p=probs)
        return top_candidates[chosen_index][0]

    def reset_buffer(self):
        self.refractory_buffer = []