# examples/run_diagnostics.py
# 診断・デバッグ・テストツール (v2.0: SaraEngine Core v48対応版)

import sys
import os
import argparse
import numpy as np
import time

# ユーティリティ
try:
    from utils import setup_path
except ImportError:
    from .utils import setup_path # type: ignore

setup_path()

try:
    # 修正: SaraGPTではなく、Coreにある SaraEngine をインポート
    from sara_engine import SaraEngine
except ImportError:
    print("Error: 'sara_engine' module not found.")
    sys.exit(1)

def run_debug_tool():
    print("=" * 60)
    print("SARA Engine Debug Tool (Core v48 Check)")
    print("=" * 60)
    
    # ダミー設定でエンジン初期化
    input_size = 20
    output_size = 2
    engine = SaraEngine(input_size, output_size)
    
    print("\n[1] Testing Initialization & Weights...")
    print("-" * 60)
    print(f"Reservoirs: {len(engine.reservoirs)} layers")
    print(f"Total Hidden Neurons: {engine.total_hidden}")
    
    # 重みの統計チェック
    w_mean = np.mean([np.mean(np.abs(w)) for w in engine.w_ho])
    print(f"Readout Weights Mean Abs: {w_mean:.4f}")
    if w_mean > 0:
        print("✓ Weights initialized correctly.")
    else:
        print("✗ Warning: Weights appear to be zero.")

    print("\n[2] Testing Sleep Phase (Adaptive Pruning)...")
    print("-" * 60)
    try:
        # 新しいシグネチャ (epoch, sample_size) のテスト
        print("Testing: sleep_phase(epoch=0, sample_size=5000) [Large Data Mode]")
        engine.sleep_phase(epoch=0, sample_size=5000)
        print("✓ Large data pruning executed successfully.")
        
        print("Testing: sleep_phase(epoch=5, sample_size=100) [Small Data Mode]")
        engine.sleep_phase(epoch=5, sample_size=100)
        print("✓ Small data pruning executed successfully.")
    except TypeError as e:
        print(f"✗ FAIL: Method signature mismatch. {e}")
        print("  Make sure core.py has sleep_phase(self, epoch, sample_size).")
        return

    print("\n[3] Testing Forward/Train Execution...")
    print("-" * 60)
    # ダミースパイク列生成 (20入力ニューロン, 5ステップ)
    dummy_spikes = []
    for _ in range(5):
        dummy_spikes.append([0, 1, 5]) # 適当な発火
    
    try:
        # 学習ステップ
        initial_pred = engine.predict(dummy_spikes)
        engine.train_step(dummy_spikes, target_label=1)
        after_pred = engine.predict(dummy_spikes)
        print(f"Prediction before/after: {initial_pred} -> {after_pred}")
        print("✓ train_step & predict executed without errors.")
    except Exception as e:
        print(f"✗ FAIL: Execution error. {e}")

    print("\nDiagnosis Complete.")

def run_learning_test():
    print("=" * 60)
    print("Learning Capability Test (Simple Binary Task)")
    print("=" * 60)
    
    engine = SaraEngine(input_size=4, output_size=2)
    
    # パターンA (前半発火) -> ラベル0
    pattern_a = [[0, 1], [0, 1], [], []]
    # パターンB (後半発火) -> ラベル1
    pattern_b = [[], [], [2, 3], [2, 3]]
    
    print("Training patterns (Pattern A->0, Pattern B->1)...")
    
    for epoch in range(10):
        # ランダム順で学習
        if np.random.random() > 0.5:
            engine.train_step(pattern_a, 0)
            engine.train_step(pattern_b, 1)
        else:
            engine.train_step(pattern_b, 1)
            engine.train_step(pattern_a, 0)
        
        # 新しい sleep_phase を適用
        engine.sleep_phase(epoch=epoch, sample_size=2)
    
    print("\nTesting Predictions:")
    pred_a = engine.predict(pattern_a)
    pred_b = engine.predict(pattern_b)
    
    print(f"Pattern A: Predicted {pred_a} (Expected 0) -> {'✓' if pred_a == 0 else '✗'}")
    print(f"Pattern B: Predicted {pred_b} (Expected 1) -> {'✓' if pred_b == 1 else '✗'}")
    
    if pred_a == 0 and pred_b == 1:
        print("\nPASS: Basic binary classification works.")
    else:
        print("\nFAIL: Classification failed.")

def main():
    parser = argparse.ArgumentParser(description="SARA Diagnostics Tool")
    parser.add_argument("mode", choices=["debug", "test"], help="Mode: debug info or learning test")
    args = parser.parse_args()
    
    if args.mode == "debug":
        run_debug_tool()
    elif args.mode == "test":
        run_learning_test()

if __name__ == "__main__":
    main()