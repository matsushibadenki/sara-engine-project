# [配置するディレクトリのパス]: ./examples/demo_mnist_snn.py
# [ファイルの日本語タイトル]: SARA-Engine MNIST手書き数字認識デモ (Gaborフィルタ V1受容野版)
# [ファイルの目的や内容]: V1単純細胞を模したGaborフィルタによる受容野を導入し、エッジ検出とパターン分離の質を生物学レベルへ引き上げる。行列演算なし・CPUのみのSNNで精度95%超えを確実なものにする。
_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_mnist_snn.py",
    "//": "タイトル: SARA-Engine MNIST手書き数字認識デモ (Gaborフィルタ V1受容野版)",
    "//": "目的: V1単純細胞を模したGaborフィルタによる受容野を導入し、エッジ検出とパターン分離の質を生物学レベルへ引き上げる。行列演算なし・CPUのみのSNNで精度95%超えを確実なものにする。"
}

import sys
import os
import time
import struct
import urllib.request
import gzip
import random
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.core.layers import DynamicLiquidLayer
from src.sara_engine.models.readout_layer import SpikeReadoutLayer

def download_mnist(filename: str, source_url: str) -> str:
    filepath = os.path.join("data", filename)
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(filepath):
        print(f"[Data] ダウンロード中: {filename} ...")
        urllib.request.urlretrieve(source_url, filepath)
        print(f"[Data] ダウンロード完了: {filename}")
    return filepath

def load_mnist_images(filepath: str) -> list[list[int]]:
    print(f"[Data] 画像を展開中: {filepath} ...")
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images: list[list[int]] = []
        for i in range(num):
            start = i * rows * cols
            end = start + rows * cols
            images.append([int(b) for b in data[start:end]])
    return images

def load_mnist_labels(filepath: str) -> list[int]:
    print(f"[Data] ラベルを展開中: {filepath} ...")
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        return [int(b) for b in data]

def prepare_mnist_dataset() -> tuple[list[list[int]], list[int], list[list[int]], list[int]]:
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

def apply_spatial_receptive_fields(layer: DynamicLiquidLayer, width: int, height: int, patch_sizes: list[int]) -> None:
    if layer.use_rust and hasattr(layer.core, 'apply_spatial_receptive_fields'):
        print(f"  -> 局所受容野（Spatial Receptive Fields: Gabor）をRustコア内に構築中... (層サイズ: {layer.size})")
        layer.core.apply_spatial_receptive_fields(width, height, patch_sizes) # type: ignore
        return

    print(f"  -> 局所受容野（Spatial Receptive Fields: Gabor）をPythonで構築中... (層サイズ: {layer.size})")
    layer.in_weights = [{} for _ in range(width * height)]
    
    angles = [0.0, math.pi / 4.0, math.pi / 2.0, math.pi * 3.0 / 4.0]
    phases = [0.0, math.pi / 2.0]
    
    for hidden_idx in range(layer.size):
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)
        patch_size = random.choice(patch_sizes)
        half_p = patch_size // 2
        
        theta = random.choice(angles)
        psi = random.choice(phases)
        lambda_val = patch_size * 0.8
        sigma = patch_size * 0.4
        gamma = 0.5
        
        for dy in range(-half_p, half_p + 1):
            for dx in range(-half_p, half_p + 1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < width and 0 <= y < height:
                    inp_idx = y * width + x
                    
                    x_prime = dx * math.cos(theta) + dy * math.sin(theta)
                    y_prime = -dx * math.sin(theta) + dy * math.cos(theta)
                    
                    env = math.exp(-(x_prime**2 + gamma**2 * y_prime**2) / (2.0 * sigma**2))
                    carrier = math.cos(2.0 * math.pi * x_prime / lambda_val + psi)
                    
                    w = env * carrier * 8.0
                    w += random.uniform(-0.5, 0.5) 
                    
                    layer.in_weights[inp_idx][hidden_idx] = w

def preprocess_images(images: list[list[int]], thresholds: list[int]) -> list[list[list[int]]]:
    print("  -> Rank-Order エンコーディングの事前計算中...")
    preprocessed: list[list[list[int]]] = []
    min_thresh = thresholds[-1]
    num_steps = len(thresholds)
    for img in images:
        active_per_step: list[list[int]] = [[] for _ in range(num_steps)]
        for idx, pixel in enumerate(img):
            if pixel > min_thresh:
                for step in range(num_steps):
                    if pixel > thresholds[step]:
                        active_per_step[step].append(idx)
                        break
        preprocessed.append(active_per_step)
    return preprocessed

def run_mnist_snn() -> None:
    print("="*60)
    print("SARA-Engine: MNIST Classification (Gabor V1 Receptive Fields)")
    print("="*60)

    train_images, train_labels, test_images, test_labels = prepare_mnist_dataset()
    
    num_inputs = 784
    num_classes = 10
    epochs = 4  # Gaborフィルタ導入により早期の収束が可能に
    
    train_samples_to_use = len(train_images)
    test_samples_to_use = len(test_images)

    print("\n[Network] 生物学的マルチスケール・ネットワークを構築中...")
    
    layer_configs = [
        {"size": 4000, "decay": 0.50, "patch_sizes": [4, 6, 8]},
        {"size": 4000, "decay": 0.80, "patch_sizes": [12, 16, 20]}
    ]
    
    liquid_layers: list[DynamicLiquidLayer] = []
    for cfg in layer_configs:
        size_val = int(cfg["size"]) # type: ignore
        decay_val = float(cfg["decay"]) # type: ignore
        patch_sizes_val = cfg["patch_sizes"] # type: ignore
        layer = DynamicLiquidLayer(
            input_size=num_inputs, 
            hidden_size=size_val, 
            decay=decay_val, 
            target_rate=0.05,
            density=0.0,
            input_scale=0.0
        )
        apply_spatial_receptive_fields(layer, 28, 28, patch_sizes_val) # type: ignore
        liquid_layers.append(layer)
    
    thresholds = [220, 160, 100, 40]
    num_steps = len(thresholds)
    
    total_hidden = sum(int(cfg["size"]) for cfg in layer_configs) # type: ignore
    feature_dim_per_step = num_inputs + total_hidden
    
    temporal_spatial_size = feature_dim_per_step * num_steps
    
    readout_layer = SpikeReadoutLayer(
        d_model=temporal_spatial_size, 
        vocab_size=num_classes,
        learning_rate=0.02,
        use_refractory=False
    )

    print("\n[Pre-process] 画像データの前処理...")
    train_encoded = preprocess_images(train_images, thresholds)
    test_encoded = preprocess_images(test_images, thresholds)

    print(f"\n[Train] 学習開始 (データ数: {train_samples_to_use}件 x {epochs}エポック)...")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        combined = list(zip(train_encoded, train_labels))
        random.shuffle(combined)
        
        # zip(*) 展開時の曖昧なジェネレータ型推論を回避するため、明示的リスト内包を使用
        train_encoded_shuffled: list[list[list[int]]] = [item[0] for item in combined]
        train_labels_shuffled: list[int] = [item[1] for item in combined]
        
        if epoch > 0:
            readout_layer.learning_rate *= 0.85
            
        for i in range(train_samples_to_use):
            encoded_image = train_encoded_shuffled[i]
            target_label = train_labels_shuffled[i]
            
            for l in liquid_layers:
                if l.use_rust and hasattr(l.core, 'reset_potentials'):
                    l.core.reset_potentials() # type: ignore
                else:
                    l.v = [0.0] * l.size
                    l.refractory = [0.0] * l.size
                    # ジェネレータ式を避けるための明示的ループ
                    new_thresh: list[float] = []
                    for t in l.dynamic_thresh:
                        new_thresh.append(5.0 if t > 5.0 else t)
                    l.dynamic_thresh = new_thresh
            
            accumulated_fired: set[int] = set()
            prev_fired: list[list[int]] = [[] for _ in range(len(liquid_layers))]
            hidden_already_fired: list[set[int]] = [set() for _ in range(len(liquid_layers))]
            
            for step, active_inputs in enumerate(encoded_image):
                if not active_inputs:
                    continue
                
                step_base_offset = step * feature_dim_per_step
                for inp in active_inputs:
                    accumulated_fired.add(inp + step_base_offset)
                
                hidden_offset = num_inputs
                for l_idx, l in enumerate(liquid_layers):
                    fired_hidden = l.forward(
                        active_inputs=active_inputs,
                        prev_active_hidden=prev_fired[l_idx]
                    )
                    prev_fired[l_idx] = fired_hidden
                    
                    first_fired: list[int] = []
                    for f in fired_hidden:
                        if f not in hidden_already_fired[l_idx]:
                            first_fired.append(f)
                            
                    if first_fired:
                        hidden_already_fired[l_idx].update(first_fired)
                        for f in first_fired:
                            accumulated_fired.add(f + step_base_offset + hidden_offset)
                        
                    hidden_offset += l.size
                
            if accumulated_fired:
                readout_layer.forward(list(accumulated_fired), target_token=target_label, learning=True)
                
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{train_samples_to_use} samples...")

    print(f"学習完了 | 所要時間: {time.time() - start_time:.2f}秒")

    print("\n[Sleep] 睡眠フェーズ開始 (ノイズシナプスの枝刈り)...")
    pruned_count = 0
    total_count = 0
    prune_rate = 0.005
    for s_idx in range(len(readout_layer.W)):
        to_delete: list[int] = []
        for t_id, weight in readout_layer.W[s_idx].items():
            if abs(weight) < prune_rate:
                to_delete.append(t_id)
                
        for t_id in to_delete:
            del readout_layer.W[s_idx][t_id]
            pruned_count += 1
            
        total_count += len(readout_layer.W[s_idx]) + len(to_delete)
        
    print(f"睡眠フェーズ完了: 全 {total_count} シナプス中 {pruned_count} 個を枝刈りしました。")

    print(f"\n[Test] 推推論テスト開始 (テストデータ数: {test_samples_to_use}件)...")
    correct_count = 0
    test_start_time = time.time()
    
    for i in range(test_samples_to_use):
        encoded_image = test_encoded[i]
        true_label = test_labels[i]
        
        for l in liquid_layers:
            if l.use_rust and hasattr(l.core, 'reset_potentials'):
                l.core.reset_potentials() # type: ignore
            else:
                l.v = [0.0] * l.size
                l.refractory = [0.0] * l.size
            
        accumulated_fired = set()
        prev_fired = [[] for _ in range(len(liquid_layers))]
        hidden_already_fired = [set() for _ in range(len(liquid_layers))]
        
        for step, active_inputs in enumerate(encoded_image):
            if not active_inputs:
                continue
            
            step_base_offset = step * feature_dim_per_step
            for inp in active_inputs:
                accumulated_fired.add(inp + step_base_offset)
            
            hidden_offset = num_inputs
            for l_idx, l in enumerate(liquid_layers):
                fired_h = l.forward(
                    active_inputs=active_inputs,
                    prev_active_hidden=prev_fired[l_idx]
                )
                prev_fired[l_idx] = fired_h
                
                first_fired = []
                for f in fired_h:
                    if f not in hidden_already_fired[l_idx]:
                        first_fired.append(f)
                        
                if first_fired:
                    hidden_already_fired[l_idx].update(first_fired)
                    for f in first_fired:
                        accumulated_fired.add(f + step_base_offset + hidden_offset)
                    
                hidden_offset += l.size
            
        if accumulated_fired:
            predicted = readout_layer.forward(list(accumulated_fired), learning=False)
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