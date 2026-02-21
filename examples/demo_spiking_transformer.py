# /Users/Shared/Program/python310/sara-engine-project/examples/demo_spiking_transformer.py
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
    seq_len = 64
    d_model = 32
    d_ff = 64
    num_layers = 2
    simulation_steps = 15
    
    # 修正: trainable=True を指定し、学習プロセスを有効化
    model = SpikingTransformer(
        vocab_size=vocab_size, 
        seq_len=seq_len, 
        d_model=d_model, 
        d_ff=d_ff, 
        num_layers=num_layers,
        trainable=True
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
        epoch_log = {"epoch": epoch + 1, "samples": []}
        
        print(f"\n--- Epoch {epoch + 1:2d} ---")
        for text in training_data:
            tokens = text_to_bytes(text)
            target_tokens = list(tokens)
            
            # Forward pass and Hebbian learning
            result = model(tokens, target_tokens=target_tokens, simulation_steps=simulation_steps, return_input_len=True)
            if isinstance(result, tuple):
                predicted_tokens, input_len = result
            else:
                predicted_tokens = result
                input_len = len(tokens)
            
            predicted_tokens = predicted_tokens[:input_len]

            output_text = bytes_to_text(predicted_tokens)
            
            sample_log = {
                "input": text,
                "input_bytes": tokens,
                "predicted_bytes": predicted_tokens,
                "predicted_text": output_text
            }
            epoch_log["samples"].append(sample_log)
            
            print(f"In:  '{text}'\nOut: '{output_text}'")
            
        logs.append(epoch_log)
        # エポック終了時に一括正規化とEMA安定化を実行
        model.flush_normalize()  # trainable=True の場合にのみ機能
        
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