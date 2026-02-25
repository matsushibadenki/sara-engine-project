_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmark_memory_retention.py",
    "//": "ファイルの日本語タイトル: 長距離依存性・記憶保持ベンチマーク",
    "//": "ファイルの目的や内容: SNNのSpikeSelfAttentionが、短期バッファ（コンテキストサイズ内）と長期記憶（STDP荷重）の両方を利用し、ノイズトークンを挟んだ長距離の依存関係を保持できるかテストする。"
}

import time
import random
from sara_engine.models.spiking_causal_lm import SpikingCausalLM, SpikingCausalLMConfig

def run_benchmark():
    print("=== SARA Engine: Long-Range Dependency & Memory Retention Benchmark ===\n")
    
    # 1. セットアップ
    # コンテキストサイズをあえて小さく(20)設定し、それを超えるノイズを挟んだ時にSTDPが機能するかを見る
    config = SpikingCausalLMConfig(vocab_size=1000, embed_dim=128, context_size=20)
    model = SpikingCausalLM(config)
    
    # テスト用のトークンID
    KEY_TOKEN = 100
    VALUE_TOKEN = 200
    NOISE_START = 300
    NOISE_END = 900
    
    # 2. 学習フェーズ (関連性の刷り込み)
    print("[*] Phase 1: Associative Learning (Key -> Value)")
    # [KEY] の次に [VALUE] が来るという因果関係をSTDPで学習させる
    # シナプス結合を確実なものにするため、数回反復して刺激を与える
    for _ in range(5):
        model.learn_sequence([KEY_TOKEN, VALUE_TOKEN])
    print(f"  -> Learned sequence: [{KEY_TOKEN}] -> [{VALUE_TOKEN}]")
    
    # 3. ノイズフェーズ (コンテキストバッファの押し出し)
    # コンテキストサイズ(20)を超える長さのランダムなトークンを学習・入力し、
    # SpikeSelfAttention の short-term buffer (key_buffer/value_buffer) からKEYの記憶を物理的に押し出す
    noise_length = 50 
    print(f"\n[*] Phase 2: Injecting {noise_length} Noise Tokens (Buffer Overwrite)")
    noise_sequence = [random.randint(NOISE_START, NOISE_END) for _ in range(noise_length)]
    
    start_time = time.time()
    model.learn_sequence(noise_sequence)
    elapsed = time.time() - start_time
    
    print(f"  -> Noise injection completed in {elapsed:.4f} seconds.")
    print(f"  -> Attention Context Buffer length after noise: {len(model.block.attention.key_buffer)}")
    
    # 4. 検索・生成フェーズ (長期記憶からの抽出)
    print("\n[*] Phase 3: Long-Range Retrieval Test")
    print(f"  -> Prompting model with KEY_TOKEN [{KEY_TOKEN}]...")
    
    # 生成テスト (プロンプトは KEY_TOKEN のみ、次に VALUE_TOKEN が引き出せるか)
    generated_ids = model.generate([KEY_TOKEN], max_length=1)
    
    print(f"\n[Result]")
    print(f"  Expected Output Token: [{VALUE_TOKEN}]")
    if generated_ids:
        print(f"  Actual Output Token  : {generated_ids}")
        if VALUE_TOKEN in generated_ids:
            print("  => SUCCESS: The model successfully retained and retrieved the long-range dependency using STDP!")
        else:
            print("  => FAILED: The model forgot the association. (STDP weights / Routing may need tuning)")
    else:
        print("  => FAILED: Model generated no tokens.")

if __name__ == "__main__":
    run_benchmark()