# examples/run_classifier.py
# 分類タスク実行スクリプト（MNIST / Text Classification）
# v2.0: Core側の適応型プルーニングに対応

import argparse
import time
import numpy as np
import os
import sys

# ユーティリティの読み込み
try:
    from utils import setup_path, img_to_poisson, text_to_spikes, load_mnist_data
except ImportError:
    from .utils import setup_path, img_to_poisson, text_to_spikes, load_mnist_data # type: ignore

# パス設定
setup_path()

try:
    from sara_engine import SaraEngine
except ImportError:
    print("Error: 'sara_engine' module not found.")
    sys.exit(1)

def run_mnist(epochs=5, samples=8000, save_path="sara_mnist_model.pkl"):
    """MNIST学習実行"""
    input_size = 784
    output_size = 10
    time_steps = 60
    
    print(f"\n=== MNIST Classification Task ===")
    print(f"Settings: Epochs={epochs}, Samples={samples}, TimeSteps={time_steps}")
    
    engine = SaraEngine(input_size, output_size)
    train_data, test_data = load_mnist_data()
    
    start_total = time.time()
    best_acc = 0.0
    
    for epoch in range(epochs):
        indices = np.random.choice(len(train_data), samples, replace=False)
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        dropout = 0.1 if epoch < 2 else 0.08
        epoch_start = time.time()
        
        for i, idx in enumerate(indices):
            img, target = train_data[idx]
            spike_train = img_to_poisson(img.numpy().flatten(), time_steps)
            engine.train_step(spike_train, target, dropout_rate=dropout)
            
            if (i+1) % 100 == 0:
                elapsed = time.time() - epoch_start
                rate = (i+1) / elapsed
                print(f"  Processed {i+1}/{samples} images ({rate:.1f} img/s)", end='\r')
        
        # 修正: Core側のロジックに任せるため、prune_rate計算を削除し引数を変更
        engine.sleep_phase(epoch=epoch, sample_size=samples)
        
        # 簡易評価
        print("  Evaluating (500 samples)...")
        correct = 0
        test_indices = np.random.choice(len(test_data), 500, replace=False)
        for idx in test_indices:
            img, target = test_data[idx]
            spike_train = img_to_poisson(img.numpy().flatten(), time_steps)
            if engine.predict(spike_train) == target:
                correct += 1
        acc = correct / 500 * 100
        print(f"  Validation Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            if save_path:
                engine.save_model(save_path)
                print(f"  Model saved to {save_path}")

    print(f"\nTotal time: {time.time() - start_total:.1f}s")

def run_text_classification(epochs=10):
    """テキスト分類実行"""
    data = [
        ("good", 0), ("great", 0), ("happy", 0), ("excellent", 0), ("yes", 0),
        ("bad", 1), ("worst", 1), ("sad", 1), ("poor", 1), ("no", 1),
        ("nice", 0), ("awful", 1)
    ]
    
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab_map = {c: i for i, c in enumerate(chars)}
    input_size = len(chars)
    output_size = 2 # 0:Positive, 1:Negative
    
    print(f"\n=== Text Classification Task ===")
    print(f"Vocab size: {input_size}, Data size: {len(data)}")
    
    engine = SaraEngine(input_size, output_size)
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(data))
        correct = 0
        
        for idx in indices:
            text, label = data[idx]
            spike_train = text_to_spikes(text, vocab_map, steps_per_char=2)
            engine.train_step(spike_train, target_label=label)
            
            # 訓練データでの精度確認
            if engine.predict(spike_train) == label:
                correct += 1
        
        accuracy = correct / len(data) * 100
        print(f"Epoch {epoch+1}: Accuracy {accuracy:.1f}%")
        
        # 修正: テキスト分類も新しいシグネチャに対応
        engine.sleep_phase(epoch=epoch, sample_size=len(data))
    
    # テスト
    print("\n--- Test Results ---")
    test_words = ["happy", "bad", "great", "awful"]
    for word in test_words:
        spikes = text_to_spikes(word, vocab_map)
        pred = engine.predict(spikes)
        result = "Positive" if pred == 0 else "Negative"
        print(f"Input: '{word}' -> {result}")

def main():
    parser = argparse.ArgumentParser(description="SARA Engine Classification Demo")
    parser.add_argument("task", choices=["mnist", "text"], help="Task to run")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--samples", type=int, default=8000, help="Samples per epoch (MNIST only)")
    parser.add_argument("--save", type=str, default="sara_model.pkl", help="Model save path")
    
    args = parser.parse_args()
    
    if args.task == "mnist":
        run_mnist(epochs=args.epochs, samples=args.samples, save_path=args.save)
    elif args.task == "text":
        run_text_classification(epochs=args.epochs)

if __name__ == "__main__":
    main()