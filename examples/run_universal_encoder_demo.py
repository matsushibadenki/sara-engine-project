_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_universal_encoder_demo.py",
    "//": "タイトル: マルチモーダル・スパイク変換デモ",
    "//": "目的: テキスト以外の連続値データをスパイク化し、可視化する。"
}

import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from sara_engine.utils.encoder import SpikeEncoder
from sara_engine.utils.visualizer import SaraVisualizer

def main():
    d_model = 256
    encoder = SpikeEncoder(d_model)
    workspace_dir = os.path.join(project_root, 'workspace', 'logs', 'encoder_demo')
    visualizer = SaraVisualizer(save_dir=workspace_dir)

    # 1. サイン波（センサーデータの模擬）をスパイク化
    steps = 100
    sine_wave = [np.sin(x / 10.0) for x in range(steps)]
    
    rate_history = []
    pop_history = []
    
    for val in sine_wave:
        # Rate Coding
        rate_history.append(encoder.rate_encode(val, min_val=-1.0, max_val=1.0))
        # Population Coding
        pop_history.append(encoder.pop_encode(val, min_val=-1.0, max_val=1.0))

    # 可視化
    visualizer.plot_raster(rate_history, title="Rate Coding (Sine Wave)", filename="rate_sine.png")
    visualizer.plot_raster(pop_history, title="Population Coding (Sine Wave)", filename="pop_sine.png")
    
    print(f"Encoder demo finished. Saved to {workspace_dir}")

if __name__ == "__main__":
    main()