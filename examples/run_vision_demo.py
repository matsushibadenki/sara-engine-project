_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_vision_demo.py",
    "//": "タイトル: 視覚SNNデモ",
    "//": "目的: 画像をスパイクに変換し、視覚野の反応を可視化する。"
}

import numpy as np
import os
import matplotlib.pyplot as plt
from sara_engine import SaraGPT, SaraVisualizer, ImageSpikeEncoder

def create_dummy_image(shape=(28, 28), pattern="cross"):
    """テスト用の単純な画像を生成"""
    img = np.zeros(shape, dtype=np.float32)
    h, w = shape
    cx, cy = w // 2, h // 2
    
    if pattern == "cross":
        # 十字
        img[cy-8:cy+8, cx-2:cx+2] = 1.0 # 縦線
        img[cy-2:cy+2, cx-8:cx+8] = 1.0 # 横線
    elif pattern == "circle":
        # 円
        y, x = np.ogrid[:h, :w]
        mask = (x - cx)**2 + (y - cy)**2 <= 8**2
        img[mask] = 1.0
        
    return img

def run_vision_demo():
    print("=== SARA Engine: Vision Demo ===")
    print("Simulating Retinal Processing...")
    
    # 1. 網膜（エンコーダ）の準備
    img_shape = (28, 28)
    encoder = ImageSpikeEncoder(shape=img_shape)
    
    # テスト画像生成
    print("Generating visual stimulus (Cross pattern)...")
    image = create_dummy_image(shape=img_shape, pattern="cross")
    
    # 2. スパイク変換 (レートコーディング)
    print("Encoding image to spikes (Rate Coding)...")
    # 明るい場所ほど激しく発火
    input_spike_train = encoder.encode_rate(image, time_steps=30, max_rate=0.8)
    
    # 3. 脳の準備
    # 視覚野として機能させるため、入力サイズを画像サイズに合わせる
    input_neurons = img_shape[0] * img_shape[1] # 784
    brain = SaraGPT(sdr_size=input_neurons)
    
    # Pythonモードで初期化
    for layer in brain.layers:
        layer.use_rust = False
        layer.__init__(
            input_size=layer.input_size, 
            hidden_size=layer.size, 
            decay=0.4,    # 視覚は残像を残すため少し遅めの減衰でも良いが、今回はキレ重視
            density=0.05, 
            input_scale=1.5,
            rec_scale=0.8,
            feedback_scale=0.3,
            use_rust=False
        )
        
    # 4. 見る (Processing)
    print("Watching...")
    spike_history_l1 = []
    
    for t, input_spikes in enumerate(input_spike_train):
        brain.forward_step(input_spikes, training=False)
        spike_history_l1.append(brain.prev_spikes[0])
        
    print("Processing complete.")
    
    # 5. 可視化
    save_dir = "workspace/vision_logs"
    os.makedirs(save_dir, exist_ok=True)
    viz = SaraVisualizer(save_dir=save_dir)
    
    print("Generating visualizations...")
    
    # 入力画像の確認
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.title("Input Stimulus")
    plt.axis('off')
    plt.savefig(f"{save_dir}/input_image.png")
    plt.close()
    
    # 網膜出力（入力スパイク）
    viz.plot_raster(input_spike_train, 
                   title="Retinal Output (Input Spikes)", 
                   filename="vision_retina_raster.png")
    
    # 視覚野の反応
    viz.plot_raster(spike_history_l1, 
                   title="Visual Cortex Response (Layer 1)", 
                   filename="vision_cortex_response.png")
    
    print(f"\nSaved visualization to '{save_dir}/'.")
    print("1. 'input_image.png': The cross pattern shown to SARA.")
    print("2. 'vision_cortex_response.png': How SARA's neurons reacted.")

if __name__ == "__main__":
    run_vision_demo()