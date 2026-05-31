_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spike_attention.py",
    "//": "ファイルの日本語タイトル: スパイクアテンションのデモ",
    "//": "ファイルの目的や内容: 質問(Query)スパイクと記憶(Key)スパイクの同期発火により、正しい回答(Value)スパイクが抽出される様子を検証する。"
}

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# 既存の機能を壊さないよう、新規作成したモジュールからインポート
from sara_engine.core.coincidence_attention import SpikeDrivenAttention

def word_to_spikes(word: str, seed_offset: int = 0) -> set[int]:
    """デモ用: 単語を疑似的なSDR(疎分散表現)スパイクセットに変換する"""
    import hashlib
    spikes: set[int] = set()
    hash_obj = hashlib.md5((word + str(seed_offset)).encode())
    hash_bytes = hash_obj.digest()
    for i in range(10):
        neuron_idx = (hash_bytes[i] + seed_offset) % 1000
        spikes.add(neuron_idx)
    return spikes

def main():
    print("=== SARA Engine: Matrix-Free Spike-Driven Attention ===")
    
    attention = SpikeDrivenAttention(context_size=10, threshold=5.0)
    
    print("\n[Phase 1]: Reading Context into Attention Memory...")
    context_pairs = [
        ("首都", "東京"),   
        ("言語", "Python"), 
        ("AI", "SARA"),     
        ("天気", "晴れ")    
    ]
    
    for key_word, value_word in context_pairs:
        q_spikes: set[int] = set()
        k_spikes = word_to_spikes(key_word)
        v_spikes = word_to_spikes(value_word)
        
        attention.forward(q_spikes, k_spikes, v_spikes)
        print(f"  Stored -> Key: '{key_word}', Value: '{value_word}'")
        
    print("\n[Phase 2]: Attending to Relevant Information...")
    
    query_word = "AI"
    print(f"  User Query: '{query_word}'")
    
    q_spikes = word_to_spikes(query_word)
    routed_value_spikes = attention.forward(q_spikes, set(), set())
    
    expected_value_spikes = word_to_spikes("SARA")
    
    overlap = len(routed_value_spikes.intersection(expected_value_spikes))
    total = len(expected_value_spikes)
    
    print("\n[Results]")
    print(f"  Target Value Spikes: {sorted(list(expected_value_spikes))[:5]}...")
    print(f"  Routed Value Spikes: {sorted(list(routed_value_spikes))[:5]}...")
    
    if overlap == total and total > 0:
        print(f"\n=> SUCCESS: Spike-Attention correctly routed the value 'SARA' without matrix multiplication!")
    else:
        print(f"\n=> FAILED: Expected {total} spikes, matched {overlap}.")
        
    print("=== Demonstration Completed ===")

if __name__ == "__main__":
    main()