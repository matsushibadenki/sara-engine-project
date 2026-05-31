# examples/demo_associative_memory_stdp.py

_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_associative_memory_stdp.py",
    "//": "ファイルの日本語タイトル: 連合記憶・パターン補完デモ (バグ修正版)",
    "//": "ファイルの目的や内容: ストリームの反復処理におけるTypeErrorを修正し、想起テストを完遂させる。"
}

import random
from sara_engine import SpikeTransformerModel, SemanticSpikeEncoder

def main():
    print("Starting Associative Memory & Pattern Completion Demo...")
    random.seed(42)
    
    embed_dim = 256
    hidden_dim = 512
    vocab_size = 100
    
    model = SpikeTransformerModel(num_layers=1, embed_dim=embed_dim, hidden_dim=hidden_dim)
    encoder = SemanticSpikeEncoder(vocab_size, embed_dim)
    
    APPLE = 10
    RED = 50
    
    print(f"\n[Phase 1] Learning Association: 'Apple' + 'Red'...")
    for epoch in range(30):
        model.reset_state()
        assoc_stream = encoder.encode_associative_stream([APPLE, RED], duration=40)
        for input_spikes in assoc_stream:
            model.forward(input_spikes, learning=True)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Deep Training...")

    print("\n[Phase 2] Recall Test: Presenting only 'Apple'...")
    model.reset_state()
    
    # リストとして取得
    apple_only_stream = encoder.encode_token_stream(APPLE, duration=250)
    
    red_ensemble = set(encoder.token_maps[RED])
    recalled_red_indices = set()
    
    print(f"Target 'Red' ensemble neurons: {red_ensemble}")
    
    # next()を使わず、リストをそのままループ処理
    for t, input_spikes in enumerate(apple_only_stream):
        # learning=False により想起ブースト(gain)を有効化
        output_spikes = model.forward(input_spikes, learning=False)
        
        recalled = red_ensemble.intersection(set(output_spikes))
        if recalled:
            recalled_red_indices.update(recalled)
            print(f"  Step {t}: BINGO! Recalled {len(recalled)} neurons. Total: {len(recalled_red_indices)}/{len(red_ensemble)}")
        
    print(f"\nFinal Statistics: {len(recalled_red_indices)} / {len(red_ensemble)} neurons recalled.")
    
    if len(recalled_red_indices) >= 1:
        print("\nSUCCESS: Pattern Completion achieved!")
    else:
        print("\nFAILED: No recall. Check the synaptic gain and leak settings in stdp.py.")

if __name__ == "__main__":
    main()