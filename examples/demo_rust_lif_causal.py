# ディレクトリパス: examples/demo_rust_lif_causal.py
# ファイルの日本語タイトル: Rust LIFコア動作テスト
# ファイルの目的や内容: Rust側で実装したLIFNetworkとCausalSynapsesが正常に動作し、Pythonから操作できるか確認する。

import time

def test_rust_core():
    print("=== Rust LIF & Causal Core Test ===")
    
    try:
        from sara_engine.sara_rust_core import LIFNetwork, CausalSynapses
        print("✅ Rustモジュール 'sara_rust_core' を正常にインポートしました。")
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        print("⚠️ 'pip install -e .' を実行してRustコアを再コンパイルしてください。")
        return

    # --- LIF Network Test ---
    print("\n--- 1. LIF Network Test ---")
    lif = LIFNetwork(decay_rate=0.8, threshold=1.5)
    
    # 1回目の入力 (1, 2, 3のスパイク) -> 電位は1.0になり、閾値(1.5)を超えないため発火しない
    spikes_t1 = lif.forward([1, 2, 3])
    print(f"T=1 発火したスパイク: {spikes_t1} (空なら正常)")
    
    # 2回目の入力 (2, 3, 4のスパイク) -> 2と3は前回の残存(0.8) + 今回(1.0) = 1.8 で発火するはず
    spikes_t2 = lif.forward([2, 3, 4])
    print(f"T=2 発火したスパイク: {spikes_t2} ([2, 3] または [3, 2]なら正常)")
    
    # --- Causal Synapses Test ---
    print("\n--- 2. Causal Synapses Performance Test ---")
    synapses = CausalSynapses(max_delay=5)
    
    history = [
        [1, 2, 3], # delay=0 (直近のスパイク)
        [4, 5],    # delay=1 (少し前のスパイク)
    ]
    
    # Rustコアでの学習速度テスト
    print("Rustコアで100,000ステップの学習を実行中...")
    start = time.time()
    for _ in range(100_000):
        # Python側でこれをやると数秒かかる処理です
        synapses.train_step(history, next_token=100, learning_rate=0.5)
    elapsed = time.time() - start
    print(f"✅ 100,000ステップの学習が {elapsed:.4f}秒 で完了しました！")
    
    # 推論（電位計算）のテスト
    potentials = synapses.calculate_potentials(history)
    target_pot = potentials.get(100, 0.0)
    print(f"トークンID 100 の予測電位: {target_pot:.2f} (0より大きければ学習成功)")
    
if __name__ == "__main__":
    test_rust_core()