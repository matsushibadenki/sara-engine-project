_FILE_INFO = {
    "//": "ディレクトリパス: examples/test_stdp_transformer.py",
    "//": "タイトル: STDP学習の検証テスト",
    "//": "目的: 特徴的な入力パターンを学習し、ニューロンの選択性が向上するかを確認する。"
}

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine.core.transformer import PlasticSpikeFFN
from sara_engine.utils.visualizer import SaraVisualizer

def main():
    d_model = 128
    d_ff = 512
    epochs = 100
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace', 'logs', 'stdp_test'))
    os.makedirs(workspace_dir, exist_ok=True)
    
    ffn = PlasticSpikeFFN(d_model, d_ff)
    visualizer = SaraVisualizer(save_dir=workspace_dir)
    
    # 固定の「学習させたいパターン」を作成
    pattern_a = [10, 20, 30, 40, 50]
    
    history = []
    
    print(f"Starting STDP Learning for {epochs} epochs...")
    for epoch in range(epochs):
        # パターンAを入力して学習させる
        fired = ffn.compute(pattern_a, learning=True)
        history.append(fired)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Fired {len(fired)} neurons")

    # 学習後の反応を確認（学習なしモード）
    final_fired = ffn.compute(pattern_a, learning=False)
    print(f"Final response to pattern A: {final_fired}")
    
    # 可視化: 学習に伴い発火パターンが固定化（自己組織化）されていく様子を確認
    visualizer.plot_raster(history, title="STDP Self-Organization Process", filename="stdp_learning_raster.png")
    
    # 膜電位の代わりとして、最後の発火時のポテンシャル分布を可視化（ダミーデータで構成）
    dummy_v = np.random.randn(d_ff) # 実際は内部ポテンシャルを取得する口が必要
    dummy_thresh = np.ones(d_ff)
    visualizer.plot_membrane_potential_distribution(dummy_v, dummy_thresh, filename="final_potential_dist.png")

    print(f"Results saved to {workspace_dir}")

if __name__ == "__main__":
    main()