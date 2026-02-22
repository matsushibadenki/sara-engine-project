_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_associative_memory_stdp.py",
    "//": "ファイルの日本語タイトル: 連合記憶・パターン補完デモ (強化版)",
    "//": "ファイルの目的や内容: 学習ステップを増やし、STDPによる概念の結びつきをより確実に検証する。"
}

import os
import random
from sara_engine import SpikeTransformerModel, SemanticSpikeEncoder

def main():
    print("Starting Associative Memory & Pattern Completion Demo...")
    random.seed(42)
    
    embed_dim = 256
    hidden_dim = 512
    vocab_size = 100
    
    # モデルの初期化
    model = SpikeTransformerModel(num_layers=1, embed_dim=embed_dim, hidden_dim=hidden_dim)
    encoder = SemanticSpikeEncoder(vocab_size, embed_dim)
    
    APPLE = 10
    RED = 50
    
    # 1. 連合学習フェーズ (繰り返し提示して結合を強化)
    print("\n[Phase 1] Learning Association: 'Apple' + 'Red'...")
    for epoch in range(5):
        model.reset_state()
        # 同時提示ストリーム
        assoc_stream = encoder.encode_associative_stream([APPLE, RED], duration=15)
        total_spikes = 0
        for input_spikes in assoc_stream:
            out = model.forward(input_spikes, learning=True)
            total_spikes += len(out)
        print(f"  Epoch {epoch}: Trained on associative stream (Total Out Spikes: {total_spikes})")
    
    # 2. 想起テストフェーズ (Appleのみ提示)
    print("\n[Phase 2] Recall Test: Presenting only 'Apple'...")
    model.reset_state()
    apple_only_stream = encoder.encode_token_stream(APPLE, duration=10)
    
    red_ensemble = set(encoder.token_maps[RED])
    recalled_red_indices = set()
    
    for t, input_spikes in enumerate(apple_only_stream):
        output_spikes = model.forward(input_spikes, learning=False)
        recalled = red_ensemble.intersection(set(output_spikes))
        recalled_red_indices.update(recalled)
        if recalled:
            print(f"  Step {t}: Recalled {len(recalled)} neurons from 'Red' ensemble!")
        
    print(f"\nTotal unique 'Red' neurons recalled: {len(recalled_red_indices)} / {len(red_ensemble)}")
    
    if len(recalled_red_indices) > 0:
        print("\nSUCCESS: Pattern Completion achieved! Apple triggered Red neurons.")
    else:
        print("\nFAILED: No recall detected. Check synaptic thresholds.")

if __name__ == "__main__":
    main()