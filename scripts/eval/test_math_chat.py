# ディレクトリパス: scripts/eval/test_math_chat.py
# ファイルの日本語タイトル: 数式・一般知識ファジー推論テストスクリプト（フォールバック対応版）
# ファイルの目的や内容: 曖昧検索の閾値を調整し、連想記憶が見つからない場合は無言にならずにMoE/LIF汎化ネットワークに推論を委ねる。

import torch
import msgpack
import os
import random
from transformers import AutoTokenizer
import sys

# SARA Engineモジュールパスを認識させる
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.sara_engine.models.spiking_llm import SpikingLLM

def run_math_chat(model_path):
    if not os.path.exists(model_path):
        print(f"❌ '{model_path}' が見つかりません。")
        return
        
    print("Initializing Advanced SNN Model with Fuzzy Recall...")
    student = SpikingLLM(num_layers=2, sdr_size=8192, vocab_size=256000)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    print(f"Loading SNN memory from: {model_path}...")
    
    if hasattr(student, "load_memory"):
        loaded_count = student.load_memory(model_path)
    else:
        with open(model_path, "rb") as f:
            state = msgpack.unpack(f, raw=False)
        raw_map = state.get("direct_map", {})
        student._direct_map = {eval(k): {int(tk): float(tv) for tk, tv in v.items()} for k, v in raw_map.items()}
        loaded_count = len(student._direct_map)
        
    print(f"✅ Loaded {loaded_count} patterns.")

    print("\n=======================================================")
    print("🤖 SARA Engine ファジー推論テスト (フォールバック対応)")
    print("=======================================================\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']: break
            
        prompt = f"User: {user_input}\nSARA:"
        context_tokens = tokenizer(prompt)["input_ids"].copy()
        print("SARA: ", end="", flush=True)
        
        confidence_printed = False
        refractory_counters = {}
        
        for _ in range(100):
            ctx = context_tokens[-24:] if len(context_tokens) > 24 else context_tokens
            current_spikes = student._encode_to_sdr(ctx)
            sdr_k = student._sdr_key(current_spikes)
            
            vocab_potentials = [0.0] * student.vocab_size
            
            # 💡 修正1: 閾値を 30% (0.30) まで下げて位置ズレに強くする
            recalled_data, overlap_ratio = student.recall(sdr_k, threshold=0.30)
            
            if recalled_data:
                if not confidence_printed and overlap_ratio < 1.0:
                    print(f"\n[💡 連想記憶発動: 一致度 {overlap_ratio*100:.1f}%] ", end="")
                    confidence_printed = True
                for tok_id, weight in recalled_data.items():
                    if tok_id < student.vocab_size:
                        vocab_potentials[tok_id] += weight * 10.0
            else:
                # 💡 修正2: 記憶がない場合は break せず、MoEとLIFを使って「考えて予測」する
                if not confidence_printed:
                    print(f"\n[🧠 汎化ネットワーク(MoE)による推論中...] ", end="")
                    confidence_printed = True
                
                lm_potentials, _ = student.forward(current_spikes, t_step=student.global_t)
                student.global_t += 1
                for i in range(student.vocab_size):
                    vocab_potentials[i] += lm_potentials[i]

            # 不応期（同じ言葉の繰り返し防止）
            for vocab_id in range(student.vocab_size):
                if refractory_counters.get(vocab_id, 0) > 0:
                    vocab_potentials[vocab_id] *= 0.1

            valid_indices = [i for i, p in enumerate(vocab_potentials) if p > 0.0]
            if not valid_indices:
                break

            valid_indices.sort(key=lambda i: vocab_potentials[i], reverse=True)
            top_k_indices = valid_indices[:5]
            top_potentials = [vocab_potentials[i] for i in top_k_indices]
            
            # 確率的なサンプリング（Temperature = 0.8）
            top_potentials = [p ** (1.0 / 0.8) for p in top_potentials]
            sum_p = sum(top_potentials)
            if sum_p <= 0.0: break
            
            probs = [p / sum_p for p in top_potentials]
            r = random.random()
            cumulative = 0.0
            next_token = top_k_indices[0]
            
            for idx, prob in zip(top_k_indices, probs):
                cumulative += prob
                if r <= cumulative:
                    next_token = idx
                    break
                
            context_tokens.append(next_token)
            text_chunk = tokenizer.decode([next_token])
            print(text_chunk, end="", flush=True)
            
            for k in list(refractory_counters.keys()):
                refractory_counters[k] -= 1
                if refractory_counters[k] <= 0:
                    del refractory_counters[k]
            refractory_counters[next_token] = 1
            
            if next_token == tokenizer.encode("\n", add_special_tokens=False)[-1] or "\n" in text_chunk:
                break
        print()

if __name__ == "__main__":
    run_math_chat("models/distilled_sara_llm.msgpack")