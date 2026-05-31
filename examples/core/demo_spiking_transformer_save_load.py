_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spiking_transformer_save_load.py",
    "//": "日本語タイトル: モデル保存・復元テストデモ (再現性強化版)",
    "//": "目的: 決定論的モードと状態リセットを組み合わせ、保存・復元後の出力完全一致を検証する。"
}

import os
import random
import json
from sara_engine.core.transformer import SpikeTransformerModel

def main():
    print("Starting Improved Serialization and Verification Demo...")
    
    workspace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    model_path = os.path.join(workspace_dir, "spiking_transformer_model.json")
    
    # 同じシードで入力を固定
    random.seed(42)
    test_input = random.sample(range(256), 25)
    
    # 1. モデルの初期化と学習
    model = SpikeTransformerModel(num_layers=2, embed_dim=256, hidden_dim=512)
    print("Training model for 5 steps (STDP Active)...")
    for step in range(5):
        model.forward(test_input, learning=True)
        
    # 2. 保存
    model.save_pretrained(model_path)
    print(f"Model saved to {model_path}")
    
    # 3. 新しいインスタンスへの読み込み
    new_model = SpikeTransformerModel(num_layers=2, embed_dim=256, hidden_dim=512)
    new_model.load_pretrained(model_path)
    
    # 4. 検証 (重要: 両方の内部状態をリセットし、決定論的推論を行う)
    print("\nVerifying outputs with learning=False and reset states...")
    model.reset_state()
    new_model.reset_state()
    
    out_original = model.forward(test_input, learning=False)
    out_loaded = new_model.forward(test_input, learning=False)
    
    print(f"  Original Model Output Spikes: {len(out_original)}")
    print(f"  Loaded Model Output Spikes:   {len(out_loaded)}")
    
    original_set = set(out_original)
    loaded_set = set(out_loaded)
    
    if original_set == loaded_set:
        print("\nSUCCESS: SNN State Serialization is now deterministic and accurate!")
    else:
        diff = original_set.symmetric_difference(loaded_set)
        print(f"\nFAILED: Output mismatch. Difference: {len(diff)} spikes.")
        # デバッグ用に一部表示
        print(f"  Original unique: {list(original_set)[:5]}")
        print(f"  Loaded unique:   {list(loaded_set)[:5]}")

if __name__ == "__main__":
    main()