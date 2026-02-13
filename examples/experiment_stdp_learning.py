_FILE_INFO = {
    "//": "ディレクトリパス: examples/experiment_stdp_learning.py",
    "//": "タイトル: STDP学習効果検証実験",
    "//": "目的: 初期入力調整とパラメータチューニング。"
}

import numpy as np
import matplotlib.pyplot as plt
import os
from sara_engine import SaraGPT, SaraVisualizer

def run_learning_experiment():
    print("=== SARA Engine: STDP Learning Experiment (v2) ===")
    print("Experiment: Repeated stimulation with LTD enabled.")
    
    # 1. モデル初期化
    sdr_size = 1024
    engine = SaraGPT(sdr_size=sdr_size)
    
    print("Initializing engine with adjusted parameters...")
    for layer in engine.layers:
        layer.use_rust = False
        layer.__init__(
            input_size=layer.input_size, 
            hidden_size=layer.size, 
            decay=layer.decay,
            density=layer.density,
            input_scale=0.8,  # [Tuning] 初期入力を弱める (Default: 1.0~2.0)
            rec_scale=layer.rec_scale,
            feedback_scale=layer.feedback_scale,
            use_rust=False,
            target_rate=0.05 # [Fix] 目標発火率を設定
        )
    
    save_dir = "workspace/stdp_learning_logs"
    os.makedirs(save_dir, exist_ok=True)
    viz = SaraVisualizer(save_dir=save_dir)
    
    # 2. 実験設定
    target_word = "sara"
    input_sdr = engine.encoder.encode(target_word)
    repetitions = 50  # [Tuning] 反復回数を増やす
    steps_per_input = 15 # [Tuning] 1回の時間を長くする
    
    activity_history = []
    first_epoch_spikes = []
    last_epoch_spikes = []
    
    print(f"\nTarget Word: '{target_word}'")
    print(f"Repetitions: {repetitions}")
    print("Running simulation...")
    
    # 3. 学習ループ
    for epoch in range(repetitions):
        epoch_spike_count = 0
        current_epoch_spikes = []
        
        for t in range(steps_per_input):
            _, all_spikes = engine.forward_step(input_sdr, training=True)
            l2_spikes = engine.prev_spikes[1]
            epoch_spike_count += len(l2_spikes)
            current_epoch_spikes.append(l2_spikes)
        
        activity_history.append(epoch_spike_count)
        print(f"Epoch {epoch+1:02d}: Total Spikes = {epoch_spike_count}")
        
        if epoch == 0:
            first_epoch_spikes = current_epoch_spikes
        elif epoch == repetitions - 1:
            last_epoch_spikes = current_epoch_spikes

        # エポック間のリセット
        for layer in engine.layers:
            layer.v.fill(0) 
            layer.refractory.fill(0)
    
    # 4. 結果の可視化
    print("\nGenerating analysis plots...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, repetitions + 1), activity_history, marker='o', linestyle='-', color='b')
    plt.title(f"Learning Curve: Neural Activity over Repetitions ('{target_word}')")
    plt.xlabel("Repetition (Epoch)")
    plt.ylabel("Total Spike Count (Layer 2)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/learning_curve.png")
    print(f"Saved: {save_dir}/learning_curve.png")
    plt.close()
    
    viz.plot_raster(first_epoch_spikes, 
                   title=f"Spike Pattern: Epoch 1 (Before Learning)", 
                   filename="raster_epoch_01.png")
    
    viz.plot_raster(last_epoch_spikes, 
                   title=f"Spike Pattern: Epoch {repetitions} (After Learning)", 
                   filename=f"raster_epoch_{repetitions}.png")
    
    # 考察
    start = activity_history[0]
    end = activity_history[-1]
    diff = end - start
    print("\n=== Experiment Result ===")
    print(f"Start Activity: {start}")
    print(f"End Activity:   {end}")
    
    # ノイズ対策: わずかな変動は無視
    threshold = start * 0.05
    
    if diff < -threshold:
        print("Observation: Activity DECREASED significantly.")
        print("解説: [成功] LTDにより回路が効率化され、無駄な発火が抑制されました（省エネ化）。")
    elif diff > threshold:
        print("Observation: Activity INCREASED significantly.")
        print("解説: LTPが優勢で、より鋭敏に反応するようになりました。")
    else:
        print("Observation: Stable activity.")
        print("解説: LTPとLTDが釣り合っているか、飽和しています。")

if __name__ == "__main__":
    run_learning_experiment()