# examples/demo_spiking_transformer.py
# スパイキング・トランスフォーマーの実行とテスト
# BP不使用のError-Driven Hebbian学習が、エポックごとにどのように収束していくか（文字化けから正しい文字列へ）を可視化します。

import os
import json
import random
import sys

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.models.spiking_transformer_stdp import SpikingTransformer

def ensure_workspace():
    workspace_dir = os.path.join(os.path.dirname(__file__), '../workspace/spiking_transformer_logs')
    os.makedirs(workspace_dir, exist_ok=True)
    return workspace_dir

def text_to_bytes(text: str) -> list:
    return list(text.encode('utf-8'))

def bytes_to_text(byte_list: list) -> str:
    try:
        clean_bytes = [b for b in byte_list if 0 < b <= 255]
        return bytes(clean_bytes).decode('utf-8', errors='replace')
    except Exception:
        return ""

def main():
    workspace_dir = ensure_workspace()
    log_file = os.path.join(workspace_dir, 'training_log.json')
    
    print("Initializing Error-Driven Spiking Transformer...")
    vocab_size = 256
    # 修正: seq_len を 64 に拡張
    # 理由: 日本語UTF-8バイト列が33バイトで旧seq_len=32を超えていた
    #       'こんにちは、SNNの世界。' = 33バイト → 末尾が切り捨てられ '。' が文字化けしていた
    seq_len = 64
    d_model = 32  # 表現力を高めるために拡張
    d_ff = 64
    num_layers = 2
    simulation_steps = 15
    
    model = SpikingTransformer(
        vocab_size=vocab_size, 
        seq_len=seq_len, 
        d_model=d_model, 
        d_ff=d_ff, 
        num_layers=num_layers
    )
    
    training_data = [
        "Hello, Spiking World!",
        "こんにちは、SNNの世界。",
        "Rust and Python SARA.",
        "Energy efficient design."
    ]
    
    logs = []
    
    print("Starting Error-Driven Hebbian & STDP Learning Process...")
    epochs = 15
    for epoch in range(epochs):
        # To show learning curve clearly, we keep the order fixed in this demo
        epoch_log = {"epoch": epoch + 1, "samples": []}
        
        print(f"\n--- Epoch {epoch + 1:2d} ---")
        for text in training_data:
            tokens = text_to_bytes(text)
            # 修正: target = tokens のコピー（autoencoder = 入力の再生）
            # 旧: target_tokens = tokens[1:] + [0]  ← next-token prediction
            #     これは位置iの発火で tokens[i+1] を学習するため 1文字シフトが発生していた
            # 新: target_tokens = list(tokens)  ← 入力をそのまま再生するよう学習
            target_tokens = list(tokens)
            
            # Forward pass inherently triggers STDP and Error-Driven local learning
            # 修正: return_input_len=True で入力長を取得し末尾ゴミを除去
            result = model(tokens, target_tokens=target_tokens, simulation_steps=simulation_steps, return_input_len=True)
            if isinstance(result, tuple):
                predicted_tokens, input_len = result
            else:
                predicted_tokens = result
                input_len = len(tokens)
            # 末尾のPADスロット出力を除去
            predicted_tokens = predicted_tokens[:input_len]

            output_text = bytes_to_text(predicted_tokens)
            target_text = bytes_to_text(target_tokens)
            
            sample_log = {
                "input": text,
                "input_bytes": tokens,
                "predicted_bytes": predicted_tokens,
                "predicted_text": output_text
            }
            epoch_log["samples"].append(sample_log)
            
            # 予測結果が正解に近づいているか視覚的に確認
            print(f"In:  '{text}'\nOut: '{output_text}'")
            
        logs.append(epoch_log)
        # エポック終了時に一括正規化（trainable=True の場合のみ有効）
        model.flush_normalize()
        
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
        
    attn_weights = model.layers[0].attention.attn_weights
    attn_log_file = os.path.join(workspace_dir, 'attention_weights.json')
    with open(attn_log_file, 'w', encoding='utf-8') as f:
        json.dump(attn_weights, f, indent=2)
        
    print(f"\nTraining logs saved to {log_file}")
    print("Spiking Transformer simulation completed successfully without BP/Matrices.")

if __name__ == "__main__":
    main()