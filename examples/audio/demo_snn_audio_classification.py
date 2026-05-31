_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_audio_classification.py",
    "//": "ファイルの日本語タイトル: SNN音声分類デモ",
    "//": "ファイルの目的や内容: FFTなしで生の音声波形(サイン波)を生成し、ノイズを混ぜた上で低音と高音をISI(スパイク間隔)で聞き分ける。"
}

import os
import sys
import math
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.auto import AutoSNNModelForAudioClassification
from sara_engine.pipelines import pipeline

def generate_sine_wave(freq: float, sample_rate: int = 8000, duration: float = 0.5) -> list[float]:
    """生の音声波形（サイン波）を生成（NumPy不使用）"""
    wave = []
    for i in range(int(sample_rate * duration)):
        wave.append(math.sin(2 * math.pi * freq * (i / sample_rate)))
    return wave

def add_noise(wave: list[float], noise_level: float = 0.3) -> list[float]:
    """波形にホワイトノイズを混ぜて難易度を上げる"""
    return [sample + random.uniform(-noise_level, noise_level) for sample in wave]

def main():
    print("=== SARA Engine: Matrix-Free SNN Audio Classification ===")
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace", "snn_audio_demo"))
    os.makedirs(workspace_dir, exist_ok=True)
    model_dir = os.path.join(workspace_dir, "snn_audio_checkpoint")
    
    model = AutoSNNModelForAudioClassification.from_pretrained(model_dir)
    id2label = {0: "Low Pitch Audio (220Hz)", 1: "High Pitch Audio (880Hz)"}
    
    classifier = pipeline("audio-classification", model=model, id2label=id2label)
    
    # 訓練データの生成 (生の波形リスト)
    # Low = 220Hz, High = 880Hz
    train_low = generate_sine_wave(220)
    train_high = generate_sine_wave(880)
    
    train_data = [
        (train_low, 0),
        (train_high, 1)
    ]
    
    print("\n[Phase 1]: Learning Audio Frequencies via Cochlear Spikes (ISI)...")
    # STDPによる数回の局所学習（FFTなどの重い変換はゼロです）
    epochs = 5
    for epoch in range(epochs):
        for wave, label_id in train_data:
            classifier.learn(wave, label_id)
    print("Training Complete.")
    
    print("\n[Phase 2]: Inference on Unseen & Noisy Audio Waves")
    # テストデータには強いノイズを混ぜる
    test_low_noisy = add_noise(generate_sine_wave(220), noise_level=0.5)
    test_high_noisy = add_noise(generate_sine_wave(880), noise_level=0.5)
    
    test_prompts = [
        ("Noisy Low Pitch (220Hz)", test_low_noisy),
        ("Noisy High Pitch (880Hz)", test_high_noisy)
    ]
    
    for name, wave in test_prompts:
        result = classifier(wave)
        print(f"\nAudio: {name}")
        print(f" -> Predicted: {result['label']}")
        
    model.save_pretrained(model_dir)
    print("\n=== Demonstration Completed ===")

if __name__ == "__main__":
    main()