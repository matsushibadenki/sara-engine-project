_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_mnist_snn.py",
    "//": "タイトル: SARA-Engine MNIST手書き数字認識デモ (TTFS完全スパース修正版)",
    "//": "目的: TTFSコーディングのバグ修正に加え、DynamicLiquidLayerの正しいインポートパス(layers.py)を反映してエラーを解消する。"
}

import sys
import os
import time
import struct
import urllib.request
import gzip
import random

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 修正箇所: 正しい実装元である layers モジュールからインポートするように変更
from src.sara_engine.core.layers import DynamicLiquidLayer
from src.sara_engine.models.readout_layer import ReadoutLayer

def download_mnist(filename, source_url):
    filepath = os.path.join("data", filename)
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(filepath):
        print(f"[Data] ダウンロード中: {filename} ...")
        urllib.request.urlretrieve(source_url, filepath)
        print(f"[Data] ダウンロード完了: {filename}")
    return filepath

def load_mnist_images(filepath):
    print(f"[Data] 画像を展開中: {filepath} ...")
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = []
        for i in range(num):
            start = i * rows * cols
            end = start + rows * cols
            images.append([int(b) for b in data[start:end]])
    return images

def load_mnist_labels(filepath):
    print(f"[Data] ラベルを展開中: {filepath} ...")
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        return [int(b) for b in data]

def prepare_mnist_dataset():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_img": ("train-images-idx3-ubyte.gz", base_url + "train-images-idx3-ubyte.gz"),
        "train_lbl": ("train-labels-idx1-ubyte.gz", base_url + "train-labels-idx1-ubyte.gz"),
        "test_img": ("t10k-images-idx3-ubyte.gz", base_url + "t10k-images-idx3-ubyte.gz"),
        "test_lbl": ("t10k-labels-idx1-ubyte.gz", base_url + "t10k-labels-idx1-ubyte.gz"),
    }
    
    paths = {k: download_mnist(v[0], v[1]) for k, v in files.items()}
    
    train_images = load_mnist_images(paths["train_img"])
    train_labels = load_mnist_labels(paths["train_lbl"])
    test_images = load_mnist_images(paths["test_img"])
    test_labels = load_mnist_labels(paths["test_lbl"])
    
    return train_images, train_labels, test_images, test_labels

def run_mnist_snn():
    print("="*60)
    print("SARA-Engine: MNIST Classification (TTFS & Multi-Scale)")
    print("="*60)

    train_images, train_labels, test_images, test_labels = prepare_mnist_dataset()
    
    num_inputs = 784
    hidden_per_layer = 1000
    num_classes = 10
    epochs = 3
    
    train_samples_to_use = len(train_images)
    test_samples_to_use = len(test_images)

    print("\n[Network] マルチスケール・ネットワークを構築中...")
    
    decays = [0.60, 0.80, 0.95]
    liquid_layers = []
    for d in decays:
        liquid_layers.append(DynamicLiquidLayer(
            input_size=num_inputs, 
            hidden_size=hidden_per_layer, 
            decay=d, 
            target_rate=0.04,
            density=0.15,      # 入力スパイク激減を補うため結合密度を上昇
            input_scale=5.0,   # 単一スパイクのエネルギーを大幅に強化
            use_rust=False
        ))
    
    total_hidden = hidden_per_layer * len(liquid_layers)
    
    readout_layer = ReadoutLayer(
        input_size=total_hidden, 
        output_size=num_classes,
        learning_rate=0.005
    )

    print(f"\n[Train] 学習開始 (データ数: {train_samples_to_use}件 x {epochs}エポック)...")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        combined = list(zip(train_images, train_labels))
        random.shuffle(combined)
        train_images_shuffled, train_labels_shuffled = zip(*combined)
        
        readout_layer.t = 0
        for o in range(readout_layer.output_size):
            for idx in readout_layer.m[o]:
                readout_layer.m[o][idx] = 0.0
                readout_layer.v[o][idx] = 0.0
        
        if epoch > 0:
            readout_layer.lr *= 0.7  
            
        for i in range(train_samples_to_use):
            image = train_images_shuffled[i]
            target_label = train_labels_shuffled[i]
            
            for l in liquid_layers:
                l.v = [0.0] * l.size
                l.refractory = [0.0] * l.size
                for j in range(l.size):
                    if l.dynamic_thresh[j] > 4.0:
                        l.dynamic_thresh[j] = 4.0
            
            accumulated_fired = set()
            prev_fired = [[] for _ in range(len(liquid_layers))]
            
            thresholds = [192, 128, 64, 16]
            has_fired_inputs = set() # 修正ポイント：発火済みピクセルを記録するセット
            
            for step, thresh in enumerate(thresholds):
                # 修正ポイント：まだ発火していないピクセルのみを抽出（TTFSコーディングの厳密化）
                active_inputs = [
                    idx for idx, pixel in enumerate(image) 
                    if pixel > thresh and idx not in has_fired_inputs
                ]
                has_fired_inputs.update(active_inputs) # 発火リストに登録
                
                offset = 0
                for l_idx, l in enumerate(liquid_layers):
                    fired_hidden = l.forward_with_feedback(
                        active_inputs=active_inputs,
                        prev_active_hidden=prev_fired[l_idx]
                    )
                    prev_fired[l_idx] = fired_hidden
                    accumulated_fired.update([f + offset for f in fired_hidden])
                    offset += l.size
                
            if accumulated_fired:
                readout_layer.train_step(list(accumulated_fired), target_label)
                
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{train_samples_to_use} samples...")

    print(f"学習完了 | 所要時間: {time.time() - start_time:.2f}秒")

    print("\n[Sleep] 睡眠フェーズ開始 (ノイズシナプスの枝刈り)...")
    print(readout_layer.sleep_phase(prune_rate=0.01))

    print(f"\n[Test] 推論テスト開始 (テストデータ数: {test_samples_to_use}件)...")
    correct_count = 0
    test_start_time = time.time()
    
    for i in range(test_samples_to_use):
        image = test_images[i]
        true_label = test_labels[i]
        
        for l in liquid_layers:
            l.v = [0.0] * l.size
            l.refractory = [0.0] * l.size
            
        accumulated_fired = set()
        prev_fired = [[] for _ in range(len(liquid_layers))]
        
        thresholds = [192, 128, 64, 16]
        has_fired_inputs = set() # テスト時も同様に修正
        
        for step, thresh in enumerate(thresholds):
            active_inputs = [
                idx for idx, pixel in enumerate(image) 
                if pixel > thresh and idx not in has_fired_inputs
            ]
            has_fired_inputs.update(active_inputs)
            
            offset = 0
            for l_idx, l in enumerate(liquid_layers):
                fired_h = l.forward_with_feedback(
                    active_inputs=active_inputs,
                    prev_active_hidden=prev_fired[l_idx]
                )
                prev_fired[l_idx] = fired_h
                accumulated_fired.update([f + offset for f in fired_h])
                offset += l.size
            
        if accumulated_fired:
            potentials = readout_layer.predict(list(accumulated_fired))
            predicted = potentials.index(max(potentials))
            if predicted == true_label:
                correct_count += 1
                
        if (i + 1) % 2000 == 0:
            current_acc = (correct_count / (i + 1)) * 100
            print(f"  Tested {i + 1}/{test_samples_to_use} | Current Accuracy: {current_acc:.1f}%")

    final_accuracy = (correct_count / test_samples_to_use) * 100
    print("="*60)
    print(f"最終テスト精度: {final_accuracy:.2f}%")
    print(f"推論所要時間: {time.time() - test_start_time:.2f}秒")
    print("="*60)

if __name__ == "__main__":
    run_mnist_snn()