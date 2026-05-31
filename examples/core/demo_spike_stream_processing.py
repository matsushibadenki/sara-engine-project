_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_spike_stream_processing.py",
    "//": "ファイルの日本語タイトル: 時系列スパイクストリーム処理デモ (修正版)",
    "//": "ファイルの目的や内容: NameErrorを修正し、時系列スパイク信号をSNNモデルで処理するダイナミクスを検証する。"
}

import os
import random  # 追加: NameErrorの原因を解消
from sara_engine import SpikeTransformerModel, SpikeStreamDataLoader

def main():
    print("Starting Temporal Spike Stream Processing Demo...")
    
    # 実行結果やログの保存先を確保
    workspace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    embed_dim = 128
    hidden_dim = 256
    time_steps = 20  # 1つのデータに対して20ステップ（ミリ秒相当）処理
    
    # 再現性のためのシード固定
    random.seed(42)
    
    # 擬似データ：3つの異なるパターン（入力ベクトル）
    dummy_data = [
        [0.8 if i < 32 else 0.1 for i in range(embed_dim)],  # パターンA: 前半が活発
        [0.8 if i > 96 else 0.1 for i in range(embed_dim)],  # パターンB: 後半が活発
        [random.random() for _ in range(embed_dim)]         # パターンC: ランダム
    ]
    
    # モデル（1層のSNNブロック）とデータローダーの準備
    model = SpikeTransformerModel(num_layers=1, embed_dim=embed_dim, hidden_dim=hidden_dim)
    loader = SpikeStreamDataLoader(dummy_data, time_steps=time_steps)
    
    print(f"Processing {len(dummy_data)} data points over {time_steps} temporal steps each...")

    for data_idx, stream in enumerate(loader):
        print(f"\n--- Data Point {data_idx} ---")
        # データの区切りで内部の膜電位やアテンションバッファをリセット（重要）
        model.reset_state()
        
        for t, input_spikes in enumerate(stream):
            # 連続的な入力をモデルに供給し、STDP学習を有効にする
            output_spikes = model.forward(input_spikes, learning=True)
            
            # 5ステップごとにスパイクの活動状況を表示
            if t % 5 == 0:
                print(f"  t={t:02d}: Input Spikes={len(input_spikes):3d}, Output Spikes={len(output_spikes):3d}")

    print("\nStream processing completed successfully.")

if __name__ == "__main__":
    main()