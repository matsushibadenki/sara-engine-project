_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_diagnostics.py",
    "//": "タイトル: SARA エンジン診断ツール",
    "//": "目的: インストール済みのライブラリの状態を検証する。"
}

import numpy as np
from sara_engine import SaraGPT, DynamicLiquidLayer

def run_diagnosis():
    print("=" * 60)
    print("SARA Engine Diagnostics")
    print("=" * 60)
    
    # 1. パッケージのインポートチェック
    print("\n[1] Component Initialization...")
    try:
        engine = SaraGPT(sdr_size=1024)
        print(f"✓ SaraGPT initialized (SDR Size: {engine.sdr_size})")
        print(f"✓ Total Hidden Neurons: {engine.total_hidden}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # 2. Rustコアの確認
    print("\n[2] Native Acceleration Check...")
    layer = DynamicLiquidLayer(100, 200, 0.5)
    if layer.use_rust:
        print("✓ Rust Core: ACTIVE (High Performance)")
    else:
        print("! Rust Core: INACTIVE (Using Python Fallback)")

    # 3. 実行テスト
    print("\n[3] Forward Pass Test...")
    dummy_input = [1, 10, 50, 100]
    try:
        output_sdr, all_spikes = engine.forward_step(dummy_input, training=False)
        print(f"✓ Output Active Neurons: {len(output_sdr)}")
        print(f"✓ Total Spikes in Network: {len(all_spikes)}")
    except Exception as e:
        print(f"✗ Execution Error: {e}")

    print("\nDiagnosis Complete.")

if __name__ == "__main__":
    run_diagnosis()