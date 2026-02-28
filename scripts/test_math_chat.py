# ディレクトリパス: scripts/test_math_chat.py
# ファイルの日本語タイトル: 数式学習確認用チャットスクリプト
# ファイルの目的や内容: 学習した数式の知識（LaTeXと自然言語の結びつき）が正しく引き出せるかを対話形式で確認する。

import torch
import msgpack
import os
from transformers import AutoTokenizer
from sara_engine.models.spiking_llm import SpikingLLM

def run_math_chat():
    model_path = "models/distilled_sara_llm.msgpack"
    
    if not os.path.exists(model_path):
        print(f"❌ '{model_path}' が見つかりません。先に学習を行ってください。")
        return
        
    print("Initializing SNN Model (8192 neurons)...")
    student = SpikingLLM(num_layers=2, sdr_size=8192, vocab_size=256000)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    print(f"Opening SNN memory file: {model_path}...")
    with open(model_path, "rb") as f:
        state = msgpack.unpack(f, raw=False)
    
    raw_map = state.get("direct_map", {})
    fixed_map = {}
    for k, v in raw_map.items():
        fixed_map[eval(k)] = {int(tk): float(tv) for tk, tv in v.items()}
    student._direct_map = fixed_map
    print(f"✅ Loaded {len(fixed_map)} patterns.")

    print("\n=======================================================")
    print("🤖 SARA Engine 数式学習テストチャットへようこそ！")
    print("終了するには 'quit' または 'exit' と入力してください。")
    print("=======================================================\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        prompt = f"You: {user_input}\nSARA:"
        
        # SNNモデルからの推論
        input_ids = tokenizer(prompt)["input_ids"]
        context_tokens = input_ids.copy()
        
        print("SARA: ", end="", flush=True)
        
        generated_tokens = []
        for _ in range(100):  # 最大100トークン生成
            # 直近の8トークンをコンテキストとして使用
            ctx = context_tokens[-8:] if len(context_tokens) > 8 else context_tokens
            sdr_k = student._sdr_key(student._encode_to_sdr(ctx))
            
            if sdr_k in student._direct_map and student._direct_map[sdr_k]:
                # 最も重みの高いトークンを選択
                next_token = max(student._direct_map[sdr_k].items(), key=lambda x: x[1])[0]
            else:
                # 未知のパターンの場合は推論終了
                break
                
            generated_tokens.append(next_token)
            context_tokens.append(next_token)
            
            # トークンをデコードして順次表示
            text_chunk = tokenizer.decode([next_token])
            print(text_chunk, end="", flush=True)
            
            # 改行が生成されたら回答の区切りとみなす
            if next_token == tokenizer.encode("\n", add_special_tokens=False)[-1] or "\n" in text_chunk:
                break
                
        print()

if __name__ == "__main__":
    run_math_chat()