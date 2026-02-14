_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_mini_sara_lm.py",
    "//": "タイトル: SNN Mini-LM テキスト学習デモ",
    "//": "目的: 文字のシーケンスをSNNで学習し、次トークンの予測をシミュレーションする。"
}

import os
import sys
import numpy as np

# プロジェクトルートのsrcを優先的に読み込む
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from sara_engine.core.transformer import PlasticTransformerBlock
from sara_engine.utils.visualizer import SaraVisualizer

def text_to_spikes(text: str, d_model: int):
    """文字ごとに固定のスパイクパターンを生成（簡易トークナイザー）"""
    spikes_seq = []
    for char in text:
        # 文字コードをシードにして再現性を確保
        rng = np.random.RandomState(ord(char))
        spikes = rng.choice(d_model, max(1, int(d_model * 0.05)), replace=False).tolist()
        spikes_seq.append(spikes)
    return spikes_seq

def main():
    d_model = 256
    workspace_dir = os.path.join(project_root, 'workspace', 'logs', 'mini_lm')
    os.makedirs(workspace_dir, exist_ok=True)
    
    model = PlasticTransformerBlock(d_model=d_model, num_heads=4)
    visualizer = SaraVisualizer(save_dir=workspace_dir)
    
    # ターゲット: 「SARA」という繰り返しのパターン
    train_text = "SARASARASARA"
    input_data = text_to_spikes(train_text, d_model)
    
    all_history = []
    print(f"Training SNN Mini-LM on sequence: {train_text}")
    
    # 学習ループ
    for epoch in range(20):
        model.reset()
        epoch_spikes = []
        for t, spikes in enumerate(input_data):
            # 前のステップの出力は考慮せず、シーケンスの順序を学習
            out = model.compute(spikes, pos=t, learning=True)
            epoch_spikes.append(out)
        all_history.extend(epoch_spikes)
        if epoch % 5 == 0:
            print(f"Epoch {epoch} completed...")

    # 可視化
    visualizer.plot_raster(all_history, title="Mini-LM STDP Learning Process", filename="mini_lm_raster.png")
    print(f"Visualizations saved to {workspace_dir}")

if __name__ == "__main__":
    main()