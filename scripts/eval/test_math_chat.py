# ディレクトリパス: scripts/test_math_chat.py
# ファイルの日本語タイトル: 数式学習確認用チャットスクリプト
# ファイルの目的や内容: SNNモデルを読み込み、学習した知識が正しく引き出せるか対話形式で確認する。

import torch
import msgpack
import os
from transformers import AutoTokenizer
from sara_engine.models.spiking_llm import SpikingLLM

def run_math_chat(model_path):
    if not os.path.exists(model_path):
        print(f"❌ '{model_path}' が見つかりません。")
        return
        
    print("Initializing SNN Model (8192 neurons)...")
    student = SpikingLLM(num_layers=2, sdr_size=8192, vocab_size=256000)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    print(f"Opening SNN memory file: {model_path}...")
    with open(model_path, "rb") as f:
        state = msgpack.unpack(f, raw=False)
    
    raw_map = state.get("direct_map", {})
    student._direct_map = {eval(k): {int(tk): float(tv) for tk, tv in v.items()} for k, v in raw_map.items()}
    print(f"✅ Loaded {len(student._direct_map)} patterns.")

    print("\n=======================================================")
    print("🤖 SARA Engine テストチャットへようこそ！ (終了: quit)")
    print("=======================================================\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']: break
            
        prompt = f"You: {user_input}\nSARA:"
        context_tokens = tokenizer(prompt)["input_ids"].copy()
        print("SARA: ", end="", flush=True)
        
        for _ in range(100):
            ctx = context_tokens[-24:] if len(context_tokens) > 24 else context_tokens
            sdr_k = student._sdr_key(student._encode_to_sdr(ctx))
            
            if sdr_k in student._direct_map and student._direct_map[sdr_k]:
                next_token = max(student._direct_map[sdr_k].items(), key=lambda x: x[1])[0]
            else:
                break
                
            context_tokens.append(next_token)
            text_chunk = tokenizer.decode([next_token])
            print(text_chunk, end="", flush=True)
            
            if next_token == tokenizer.encode("\n", add_special_tokens=False)[-1] or "\n" in text_chunk:
                break
        print()