_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_cortex_learning_test.py",
    "//": "タイトル: 皮質カラムの局所学習と破滅的忘却のテスト（リカレント評価版）",
    "//": "目的: STDPによるリカレント結合の学習効果と、コンパートメントによる忘却防止を両立して確認する。"
}

import os
import sys
import random
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from sara_engine.core.cortex import CorticalColumn
except ImportError:
    print("Error: src/sara_engine/core/cortex.py が見つかりません。先にモジュールを配置してください。")
    sys.exit(1)

def generate_mock_sdr(seed_str: str, size: int = 1000, density: float = 0.05) -> List[int]:
    random.seed(seed_str)
    num_active = int(size * density)
    return random.sample(range(size), num_active)

def measure_overlap(sdr1: List[int], sdr2: List[int]) -> float:
    set1, set2 = set(sdr1), set(sdr2)
    if not set1 and not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def run_test():
    input_size = 1000
    hidden_size = 500
    compartments = ["fruit_expert", "tech_expert"]
    
    print("=== Cortical Column (皮質カラム) 初期化 ===")
    cortex = CorticalColumn(
        input_size=input_size, 
        hidden_size_per_comp=hidden_size, 
        compartment_names=compartments
    )
    
    sdr_apple = generate_mock_sdr("apple", size=input_size)
    sdr_rust = generate_mock_sdr("rust", size=input_size)
    
    print("\n--- [フェーズ1] 初期状態の反応（リカレントステップを評価） ---")
    cortex.reset_short_term_memory()
    # Step 1: 外部からの入力
    out_apple_init_t1 = cortex.forward_latent_chain(sdr_apple, [], current_context="fruit_expert", learning=False)
    # Step 2: 内部のリカレント結合による自己想起（ここが学習で変化する）
    out_apple_init_t2 = cortex.forward_latent_chain([], out_apple_init_t1, current_context="fruit_expert", learning=False)
    
    cortex.reset_short_term_memory()
    out_rust_init_t1 = cortex.forward_latent_chain(sdr_rust, [], current_context="tech_expert", learning=False)
    out_rust_init_t2 = cortex.forward_latent_chain([], out_rust_init_t1, current_context="tech_expert", learning=False)
    
    print(f"Fruit文脈 'apple' 初期リカレント発火数: {len(out_apple_init_t2)}")
    print(f"Tech文脈 'rust' 初期リカレント発火数: {len(out_rust_init_t2)}")
    
    print("\n--- [フェーズ2] Fruit文脈で 'apple' を学習 ---")
    cortex.reset_short_term_memory()
    prev_out = out_apple_init_t1
    for epoch in range(15):
        # 連続発火させてリカレント結合(STDP)を強化する
        prev_out = cortex.forward_latent_chain(sdr_apple, prev_out, current_context="fruit_expert", learning=True, reward_signal=1.0)
        
    cortex.reset_short_term_memory()
    out_apple_learned_t1 = cortex.forward_latent_chain(sdr_apple, [], current_context="fruit_expert", learning=False)
    out_apple_learned_t2 = cortex.forward_latent_chain([], out_apple_learned_t1, current_context="fruit_expert", learning=False)
    
    print(f"学習後の 'apple' リカレント発火数: {len(out_apple_learned_t2)}")
    similarity = measure_overlap(out_apple_init_t2, out_apple_learned_t2)
    print(f"初期状態との発火パターンの類似度: {similarity:.2f} (※1.0未満なら学習により変化した証拠)")
    
    print("\n--- [フェーズ3] Tech文脈で 'rust' を学習 (干渉テスト) ---")
    cortex.reset_short_term_memory()
    prev_out = out_rust_init_t1
    for epoch in range(15):
        prev_out = cortex.forward_latent_chain(sdr_rust, prev_out, current_context="tech_expert", learning=True, reward_signal=1.0)
        
    print("\n--- [フェーズ4] 破滅的忘却の検証 ---")
    cortex.reset_short_term_memory()
    out_apple_final_t1 = cortex.forward_latent_chain(sdr_apple, [], current_context="fruit_expert", learning=False)
    out_apple_final_t2 = cortex.forward_latent_chain([], out_apple_final_t1, current_context="fruit_expert", learning=False)
    
    overlap_final = measure_overlap(out_apple_learned_t2, out_apple_final_t2)
    print(f"Techタスク学習後の 'apple' の記憶保持率: {overlap_final * 100:.1f}%")
    
    if overlap_final >= 0.99 and similarity < 1.0:
        print(">> 大成功: リカレント結合に正しく学習が反映され、かつ別タスクからの干渉も完全に防げています！")
    else:
        print(">> 確認が必要: 学習が行われていないか、干渉が発生しています。")

if __name__ == "__main__":
    run_test()