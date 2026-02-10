# directory: examples/train_mnist.py
# title: MNIST Training Example with SARA Engine
# description: PyPIパッケージとしてインストールされたsara_engineを使用したMNIST学習デモ。

import sys
import os
import numpy as np
import time
import argparse

# ライブラリのインポート
# pip install sara_engine 済みであることを前提とします
try:
    from sara_engine import SaraEngine
except ImportError:
    # ローカル開発用に親ディレクトリをパスに追加する場合
    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
    try:
        from sara_engine import SaraEngine
    except ImportError:
        print("Error: 'sara_engine' module not found. Please install it via pip or check your path.")
        sys.exit(1)

# データセット用 (サンプルのためtorchvisionを使用しますが、エンジン自体は依存しません)
try:
    import torch
    from torchvision import datasets, transforms
except ImportError:
    print("This example requires 'torch' and 'torchvision' to load MNIST dataset.")
    print("pip install torch torchvision")
    sys.exit(1)

def get_mnist_data():
    """MNISTデータをダウンロードして正規化する"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # スクリプト実行場所のdataフォルダに保存
    data_path = os.path.join(os.path.dirname(__file__), './data')
    train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_path, train=False, transform=transform)
    return train, test

def img_to_poisson(img_flat, time_steps=50):
    """画像をポアソンスパイク列に変換（レートコーディング）"""
    img_flat = np.maximum(0, img_flat)
    img_flat = img_flat / (np.max(img_flat) + 1e-6)
    rate = img_flat * 0.4
    spike_train = []
    for _ in range(time_steps):
        # 乱数とピクセル強度を比較して発火判定
        fired = np.where(np.random.rand(len(img_flat)) < rate)[0].tolist()
        spike_train.append(fired)
    return spike_train

def evaluate(engine, dataset, n_samples=300, steps=50):
    """精度評価"""
    correct = 0
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    for idx in indices:
        img, target = dataset[idx]
        spike_train = img_to_poisson(img.numpy().flatten(), steps)
        pred = engine.predict(spike_train)
        if pred == target:
            correct += 1
    return correct / n_samples * 100

def main():
    parser = argparse.ArgumentParser(description="SARA Engine MNIST Training Example")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--samples", type=int, default=5000, help="Samples per epoch (default: 5000 for quick test)")
    parser.add_argument("--save", type=str, default="sara_mnist_model.pkl", help="Path to save the model")
    args = parser.parse_args()

    input_size = 784
    output_size = 10
    time_steps = 50
    
    print(f"Initializing SARA Engine (Liquid Harmony)...")
    print(f"Settings: Epochs={args.epochs}, Samples={args.samples}")
    
    engine = SaraEngine(input_size, output_size)
    
    print("Loading MNIST data...")
    train_data, test_data = get_mnist_data()
    
    start_total = time.time()
    
    for epoch in range(args.epochs):
        indices = np.random.choice(len(train_data), args.samples, replace=False)
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        epoch_start = time.time()
        for i, idx in enumerate(indices):
            img, target = train_data[idx]
            spike_train = img_to_poisson(img.numpy().flatten(), time_steps)
            
            # 学習ステップ
            engine.train_step(spike_train, target, dropout_rate=0.1)
            
            if (i+1) % 100 == 0:
                elapsed = time.time() - epoch_start
                rate = (i+1) / elapsed
                print(f"  Processed {i+1}/{args.samples} images ({rate:.1f} img/s)", end='\r')
        
        print(f"\n  [Sleep Phase] Optimizing connections...")
        engine.sleep_phase(prune_rate=0.05)
        
        print("  Evaluating...")
        val_acc = evaluate(engine, test_data, n_samples=500, steps=time_steps)
        print(f"  Epoch {epoch+1} Validation Accuracy: {val_acc:.2f}%")

    total_time = time.time() - start_total
    print(f"\nTraining Finished in {total_time:.1f}s")
    
    print("Running Final Evaluation (1000 samples)...")
    final_acc = evaluate(engine, test_data, n_samples=1000, steps=time_steps)
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    
    if args.save:
        engine.save_model(args.save)
        print(f"Model saved to {args.save}")

if __name__ == "__main__":
    main()