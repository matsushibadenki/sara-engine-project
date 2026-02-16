_FILE_INFO = {
    "//": "ディレクトリパス: examples/test_stdp_transformer.py",
    "//": "タイトル: STDP学習の検証テスト",
    "//": "目的: 特徴的な入力パターンを学習し、ニューロンの選択性が向上するかを確認する。"
}

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine.core.transformer import IntPlasticSpikeFFN
from sara_engine.utils.visualizer import SaraVisualizer

def main():
    d_model = 128
    d_ff = 512
    epochs = 100
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace', 'logs', 'stdp_test'))
    os.makedirs(workspace_dir, exist_ok=True)
    
    ffn = IntPlasticSpikeFFN(d_model, d_ff)
    visualizer = SaraVisualizer(save_dir=workspace_dir)
    
    pattern_a = [10, 20, 30, 40, 50]
    
    history = []
    
    print(f"Starting STDP Learning for {epochs} epochs...")
    for epoch in range(epochs):
        fired = ffn.compute(pattern_a, learning=True)
        history.append(fired)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Fired {len(fired)} neurons")

    final_fired = ffn.compute(pattern_a, learning=False)
    print(f"Final response to pattern A: {final_fired}")
    
    visualizer.plot_raster(history, title="STDP Self-Organization Process", filename="stdp_learning_raster.png")
    
    dummy_v = np.random.randn(d_ff) 
    dummy_thresh = np.ones(d_ff)
    visualizer.plot_membrane_potential_distribution(dummy_v, dummy_thresh, filename="final_potential_dist.png")

    print(f"Results saved to {workspace_dir}")

if __name__ == "__main__":
    main()