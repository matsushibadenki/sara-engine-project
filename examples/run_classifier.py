_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_classifier.py",
    "//": "タイトル: 分類タスク実行スクリプト",
    "//": "目的: MNISTやテキスト分類を行うデモ。"
}

import argparse
import time
import numpy as np
from sara_engine import SaraGPT
from utils import img_to_poisson, text_to_spikes, load_mnist_data

def run_mnist(epochs=3, samples=1000):
    print(f"\n=== MNIST Task (Samples={samples}) ===")
    
    # SaraGPTを分類器として使用
    engine = SaraGPT(sdr_size=784) # 28x28
    train_data, test_data = load_mnist_data()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        indices = np.random.choice(len(train_data), samples, replace=False)
        for i, idx in enumerate(indices):
            img, target = train_data[idx]
            # ポアソンスパイクに変換
            spike_train = img_to_poisson(img.numpy().flatten(), time_steps=20)
            
            # 学習ステップ（画像全体を順次流し込む）
            for step_spikes in spike_train:
                engine.forward_step(step_spikes, training=True)
                
            if (i+1) % 100 == 0:
                print(f"  Processed {i+1}/{samples} images", end='\r')
        print()

def run_text_task():
    print("\n=== Text Classification Task ===")
    engine = SaraGPT(sdr_size=128)
    data = [("happy", 0), ("sad", 1), ("great", 0), ("bad", 1)]
    vocab = {"happy": 0, "sad": 1, "great": 2, "bad": 3}
    
    for epoch in range(50):
        for text, label in data:
            spikes = text_to_spikes(text, vocab)
            for step in spikes:
                engine.forward_step(step, training=True)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["mnist", "text"])
    args = parser.parse_args()
    
    if args.task == "mnist":
        run_mnist()
    else:
        run_text_task()

if __name__ == "__main__":
    main()