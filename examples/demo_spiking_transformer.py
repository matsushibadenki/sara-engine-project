_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spiking_transformer.py",
    "//": "タイトル: スパイクトランスフォーマーのデモ",
    "//": "目的: 構築したSNNベースのTransformerモデルを実行し、学習と推論のログをワークスペースに出力する。"
}

import os
import json
from sara_engine.core.transformer import SpikeTransformer

def main():
    print("Starting Spiking Transformer Demo...")
    
    # 1. Setup workspace for logs
    workspace_dir = os.path.join(os.getcwd(), "workspace", "logs")
    os.makedirs(workspace_dir, exist_ok=True)
    log_file_path = os.path.join(workspace_dir, "spiking_transformer_log.json")
    
    # 2. Initialize Model (d_model=128, 2 layers, 4 heads)
    vocab_size = 10000
    model = SpikeTransformer(vocab_size=vocab_size, d_model=128, num_layers=2, num_heads=4)
    
    # Simulate a multi-lingual sentence mapped to token IDs
    # e.g., "こんにちは" (JP), "Hello" (EN), "World" (EN)
    sample_sentence_ids = [4021, 150, 892] 
    
    logs = {
        "event": "Spike Transformer Execution",
        "parameters": {
            "d_model": 128,
            "num_layers": 2,
            "num_heads": 4
        },
        "results": []
    }

    # 3. Forward Pass (Learning Mode)
    print(f"Processing sequence: {sample_sentence_ids} (Learning=True)")
    outputs_learn = model.compute(sample_sentence_ids, learning=True)
    
    for pos, spikes in enumerate(outputs_learn):
        logs["results"].append({
            "step": pos,
            "token_id": sample_sentence_ids[pos],
            "active_neurons_count": len(spikes),
            "sample_spikes": spikes[:10]  # Only log first 10 for readability
        })
        print(f" Step {pos}: Token {sample_sentence_ids[pos]} -> {len(spikes)} active neurons.")
        
    # 4. Save results to workspace
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)
        
    print(f"Demo completed. Logs saved to: {log_file_path}")

if __name__ == "__main__":
    main()