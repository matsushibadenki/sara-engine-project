_FILE_INFO = {
    "//": "ディレクトリパス: examples/train_absolute_pitch.py",
    "//": "タイトル: 絶対音感トレーニング (Self-Organization of Auditory Cortex)",
    "//": "目的: ランダムな音階入力とSTDPにより、周波数選択性を持つニューロンを育成する。"
}

import numpy as np
import os
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from sara_engine import SaraGPT, AudioSpikeEncoder

def train_absolute_pitch():
    print("=== SARA Engine: Absolute Pitch Training ===")
    print("Task: Self-organization of frequency-selective neurons via STDP.")
    
    # 1. 音階の定義 (C Major Scale)
    notes = {
        "Do (C4)": 261.6,
        "Re (D4)": 293.7,
        "Mi (E4)": 329.6,
        "Fa (F4)": 349.2,
        "So (G4)": 392.0,
        "La (A4)": 440.0,
        "Ti (B4)": 493.9
    }
    note_names = list(notes.keys())
    
    # 2. エンコーダと脳の初期化
    encoder = AudioSpikeEncoder(num_neurons=1024, sample_rate=44100)
    sdr_size = 1024
    brain = SaraGPT(sdr_size=sdr_size)
    
    # Pythonモードで初期化 (内部状態の分析用)
    print("Initializing Auditory Cortex...")
    for layer in brain.layers:
        layer.use_rust = False
        layer.__init__(
            input_size=layer.input_size, 
            hidden_size=layer.size, 
            decay=0.6,    # 音声向けに少し速めの減衰
            density=0.05, 
            input_scale=1.2,
            rec_scale=0.8,
            feedback_scale=0.2,
            use_rust=False
        )
    
    # 3. トレーニング設定
    epochs = 50           # トレーニング回数（全音階をランダムに聞くセット数）
    notes_per_epoch = 10  # 1エポックあたりに聞く音の数
    duration = 0.15       # 1つの音の長さ(秒)
    
    print(f"Training for {epochs} epochs...")
    
    # --- 学習フェーズ ---
    for epoch in range(epochs):
        # ランダムに音を選んで聞かせる
        selected_notes = random.choices(note_names, k=notes_per_epoch)
        
        for note_name in selected_notes:
            freq = notes[note_name]
            
            # 音生成 -> エンコード
            # (毎回生成することで位相やノイズに微妙な変化を持たせる効果も期待)
            wave = encoder.generate_tone(freq, duration)
            spike_train = encoder.encode_signal(wave)
            
            # 順伝播 (Learning ON)
            # 音を聞いている間、脳は学習し続ける
            for input_spikes in spike_train:
                brain.forward_step(input_spikes, training=True)
                
        # エポックごとのリセット（睡眠/休息に相当）
        # 電位はリセットするが、学習した重みは維持
        for layer in brain.layers:
            layer.v.fill(0)
            layer.refractory.fill(0)
            
        print(f"Epoch {epoch+1}/{epochs} complete.", end='\r')
        
    print("\nTraining complete. Starting evaluation...")
    
    # --- 評価フェーズ (テスト) ---
    # 各音に対するニューロンの反応を記録
    # neuron_preference[neuron_id] = { "Do": count, "Re": count, ... }
    neuron_responses = defaultdict(lambda: defaultdict(int))
    
    # 評価用に各音を1回ずつ長めに聞かせる
    test_duration = 0.3
    
    for note_name in note_names:
        freq = notes[note_name]
        print(f"Testing response to {note_name}...")
        
        wave = encoder.generate_tone(freq, test_duration)
        spike_train = encoder.encode_signal(wave)
        
        # 脳の状態をリセットしてテスト
        for layer in brain.layers:
            layer.v.fill(0)
            layer.refractory.fill(0)
            
        # 学習なしで聞く
        for input_spikes in spike_train:
            _, _ = brain.forward_step(input_spikes, training=False)
            
            # Layer 2 (Liquid Layer) の発火を記録
            # どのニューロンがこの音に反応したか？
            fired_neurons = brain.prev_spikes[1]
            for nid in fired_neurons:
                neuron_responses[nid][note_name] += 1

    # --- 分析と可視化 ---
    print("Analyzing neuron selectivity...")
    
    # 「特定の音にだけ強く反応するニューロン」を探す
    # (Selectivity Indexが高いものトップNを表示)
    
    save_dir = "workspace/pitch_training_logs"
    os.makedirs(save_dir, exist_ok=True)
    
    # ヒートマップ用データの作成
    # 行: ニューロン (Top 50 most active), 列: 音階(Do~Ti)
    
    # 総発火数が多いニューロン順にソート
    active_neurons = sorted(neuron_responses.keys(), 
                          key=lambda nid: sum(neuron_responses[nid].values()), 
                          reverse=True)
    
    top_neurons = active_neurons[:50] # 上位50個
    heatmap_data = np.zeros((len(top_neurons), len(note_names)))
    
    for i, nid in enumerate(top_neurons):
        for j, note in enumerate(note_names):
            heatmap_data[i, j] = neuron_responses[nid][note]
            
    # 正規化 (各ニューロンの最大発火数で割る -> 選択性を強調)
    row_max = heatmap_data.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    heatmap_data_norm = heatmap_data / row_max
    
    # プロット
    plt.figure(figsize=(10, 12))
    plt.imshow(heatmap_data_norm, aspect='auto', cmap='viridis', interpolation='nearest')
    
    plt.title("Neuron Selectivity Map (Tuning Curves)")
    plt.xlabel("Musical Notes")
    plt.ylabel("Neuron ID (Sorted by Activity)")
    plt.xticks(ticks=range(len(note_names)), labels=[n.split()[0] for n in note_names])
    plt.colorbar(label="Normalized Response")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/tuning_curves.png")
    print(f"Saved tuning curve map to {save_dir}/tuning_curves.png")
    
    # 結果のテキスト表示 (上位5個)
    print("\n--- Top 5 Selective Neurons ---")
    for i in range(min(5, len(top_neurons))):
        nid = top_neurons[i]
        res = neuron_responses[nid]
        # 最も反応した音
        best_note = max(res, key=res.get)
        count = res[best_note]
        total = sum(res.values())
        selectivity = count / total if total > 0 else 0
        
        print(f"Neuron #{nid:04d}: Loves '{best_note}' (Selectivity: {selectivity*100:.1f}%)")

if __name__ == "__main__":
    train_absolute_pitch()