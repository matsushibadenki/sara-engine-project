_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_multimodal_association.py",
    "//": "タイトル: マルチモーダル連想実験 (Final)",
    "//": "目的: コアに実装されたホメオスタシス機能を用いて、安定した連想記憶を実現する。"
}

import numpy as np
import os
import matplotlib.pyplot as plt
from sara_engine import SaraGPT, ImageSpikeEncoder, AudioSpikeEncoder

def create_visual_pattern(shape_name):
    img = np.zeros((28, 28), dtype=np.float32)
    cx, cy = 14, 14
    if shape_name == "circle":
        y, x = np.ogrid[:28, :28]
        mask = (x - cx)**2 + (y - cy)**2 <= 8**2
        img[mask] = 1.0
    elif shape_name == "cross":
        img[cy-8:cy+8, cx-2:cx+2] = 1.0
        img[cy-2:cy+2, cx-8:cx+8] = 1.0
    return img

def run_multimodal_experiment():
    print("=== SARA Engine: Multimodal Association (Core Homeostasis) ===")
    
    vision_size = 784
    audio_size = 1024
    total_input_size = vision_size + audio_size
    
    brain = SaraGPT(sdr_size=total_input_size)
    
    print("Initializing brain (relying on core homeostasis)...")
    for layer in brain.layers:
        layer.use_rust = False
        layer.__init__(
            input_size=layer.input_size, 
            hidden_size=layer.size, 
            decay=0.4,
            density=0.05,
            input_scale=1.2, # 刺激はしっかり与える
            rec_scale=0.5,
            feedback_scale=0.2,
            use_rust=False,
            target_rate=0.04 # [New] 目標発火率を4%に設定（スパース性を強制）
        )

    vis_encoder = ImageSpikeEncoder(shape=(28, 28))
    aud_encoder = AudioSpikeEncoder(num_neurons=audio_size, sample_rate=44100)

    print("Preparing stimuli...")
    img_circle = create_visual_pattern("circle")
    snd_low = aud_encoder.generate_tone(300.0, 0.1)
    spikes_vis_a = vis_encoder.encode_rate(img_circle, time_steps=12)
    spikes_aud_a = aud_encoder.encode_signal(snd_low)
    
    img_cross = create_visual_pattern("cross")
    snd_high = aud_encoder.generate_tone(2000.0, 0.1)
    spikes_vis_b = vis_encoder.encode_rate(img_cross, time_steps=12)
    spikes_aud_b = aud_encoder.encode_signal(snd_high)

    print("Training Association (STDP ON)...")
    epochs = 40
    
    # 手動キャリブレーションやウォームアップは不要
    # 学習中に勝手に閾値が最適化されていく
    
    for ep in range(epochs):
        # Pair A
        for t in range(len(spikes_vis_a)):
            vis = spikes_vis_a[t]
            aud = spikes_aud_a[t] if t < len(spikes_aud_a) else []
            aud = [i + vision_size for i in aud]
            brain.forward_step(vis + aud, training=True)
        
        # 休息 (この間に過活動なら閾値が上がり、沈黙なら下がる)
        for _ in range(5): brain.forward_step([], training=True)

        # Pair B
        for t in range(len(spikes_vis_b)):
            vis = spikes_vis_b[t]
            aud = spikes_aud_b[t] if t < len(spikes_aud_b) else []
            aud = [i + vision_size for i in aud]
            brain.forward_step(vis + aud, training=True)
            
        for _ in range(5): brain.forward_step([], training=True)
        
        print(f"  Epoch {ep+1}/{epochs}", end='\r')
    print("\nTraining complete.")
    
    # 評価用に、純粋な音への反応マップを作成
    print("\nMapping Auditory Neurons for Evaluation...")
    brain.reset_state()
    active_neurons_low = set()
    for t in range(len(spikes_aud_a)):
        aud_idx = [i + vision_size for i in spikes_aud_a[t]]
        _, _ = brain.forward_step(aud_idx, training=False)
        active_neurons_low.update(brain.prev_spikes[1])
        
    brain.reset_state()
    active_neurons_high = set()
    for t in range(len(spikes_aud_b)):
        aud_idx = [i + vision_size for i in spikes_aud_b[t]]
        _, _ = brain.forward_step(aud_idx, training=False)
        active_neurons_high.update(brain.prev_spikes[1])

    print("\nTesting Recall (Vision ONLY)...")
    
    # Test A
    brain.reset_state()
    recall_score_low = 0
    recall_score_high = 0
    for t in range(len(spikes_vis_a)):
        vis = spikes_vis_a[t]
        _, _ = brain.forward_step(vis, training=False)
        fired = set(brain.prev_spikes[1])
        recall_score_low += len(fired.intersection(active_neurons_low))
        recall_score_high += len(fired.intersection(active_neurons_high))
        
    diff_a = recall_score_low - recall_score_high
    print(f"\n[Stimulus: Circle]")
    print(f"  -> Recall Low: {recall_score_low}, High: {recall_score_high}")
    print(f"  -> Result: {'SUCCESS' if diff_a > 0 else 'FAIL'} (Diff: {diff_a})")

    # Test B
    brain.reset_state()
    recall_score_low_2 = 0
    recall_score_high_2 = 0
    for t in range(len(spikes_vis_b)):
        vis = spikes_vis_b[t]
        _, _ = brain.forward_step(vis, training=False)
        fired = set(brain.prev_spikes[1])
        recall_score_low_2 += len(fired.intersection(active_neurons_low))
        recall_score_high_2 += len(fired.intersection(active_neurons_high))

    diff_b = recall_score_high_2 - recall_score_low_2
    print(f"\n[Stimulus: Cross]")
    print(f"  -> Recall Low: {recall_score_low_2}, High: {recall_score_high_2}")
    print(f"  -> Result: {'SUCCESS' if diff_b > 0 else 'FAIL'} (Diff: {diff_b})")
    
    # グラフ保存
    save_dir = "multimodal_logs"
    os.makedirs(save_dir, exist_ok=True)
    labels = ['Stim: Circle', 'Stim: Cross']
    low_scores = [recall_score_low, recall_score_low_2]
    high_scores = [recall_score_high, recall_score_high_2]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, low_scores, width, label='Recall: Low Tone', color='blue')
    ax.bar(x + width/2, high_scores, width, label='Recall: High Tone', color='red')
    ax.set_ylabel('Spike Overlap')
    ax.set_title('Cross-Modal Association (Core Homeostasis)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(f"{save_dir}/association_result.png")
    print(f"Saved chart to {save_dir}/association_result.png")

if __name__ == "__main__":
    run_multimodal_experiment()