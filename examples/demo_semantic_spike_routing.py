_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_semantic_spike_routing.py",
    "//": "ファイルの日本語タイトル: 意味的スパイク・ルーティング・デモ",
    "//": "ファイルの目的や内容: 概念の再出現時に、SNNのAttention機構がどう反応するかを検証する。"
}

import os
import random
from sara_engine import SpikeTransformerModel, SemanticSpikeEncoder

def main():
    print("Starting Semantic Spike Routing Demo...")
    random.seed(42)
    
    embed_dim = 256
    hidden_dim = 512
    vocab_size = 100
    
    model = SpikeTransformerModel(num_layers=2, embed_dim=embed_dim, hidden_dim=hidden_dim)
    encoder = SemanticSpikeEncoder(vocab_size, embed_dim)
    
    # シナリオ: Apple(10) -> Banana(20) -> Apple(10)
    tokens = [10, 20, 10]
    
    print(f"Processing sequence: {tokens}")
    model.reset_state()
    
    for i, token in enumerate(tokens):
        print(f"\n--- Presenting Token {token} (Index {i}) ---")
        token_stream = encoder.encode_token_stream(token, duration=10)
        
        total_out_spikes = 0
        for t, input_spikes in enumerate(token_stream):
            # STDPを有効にして処理
            output_spikes = model.forward(input_spikes, learning=True)
            total_out_spikes += len(output_spikes)
            
            if t == 9:
                print(f"  Step 9 Output: {len(output_spikes)} spikes")
        
        print(f"  Avg Activity: {total_out_spikes / 10:.1f}")

    print("\nDemo completed.")

if __name__ == "__main__":
    main()